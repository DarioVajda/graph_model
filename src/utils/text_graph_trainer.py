import torch
from transformers import Trainer

class GraphTrainer(Trainer):
    def __init__(self, *args, custom_prediction_step=None, **kwargs):
        # 1. FORCE KEEP UNUSED COLUMNS
        # We intercept the TrainingArguments during initialization to ensure 
        # the Trainer never deletes our custom 'input_graph_batch' data.
        if "args" in kwargs and kwargs["args"] is not None:
            kwargs["args"].remove_unused_columns = False

        self.custom_prediction_step = custom_prediction_step
            
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        How the loss is computed by Trainer.
        """
        # EXTRACT LABELS
        labels = inputs.pop("labels", None)
        if labels is None:
            raise ValueError("Labels must be provided in the inputs for loss computation.")
        
        # FORWARD PASS
        outputs = model(
            input_ids=None, 
            input_graph_batch=inputs, 
            labels=labels, 
            **kwargs
        )
        
        # EXTRACT LOSS
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs[0]

        # NORMALIZE LOSS BY GRADIENT ACCUMULATION STEPS
        if self.args.gradient_accumulation_steps > 1 and self.model.training:
            loss = loss / self.args.gradient_accumulation_steps
        
        return (loss, outputs) if return_outputs else loss

    def floating_point_ops(self, inputs):
        """
        Bypasses the default Hugging Face FLOPs calculation which crashes 
        when trying to call .numel() on our custom graph lists.
        """
        return 0

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if self.custom_prediction_step is not None:
            return self.custom_prediction_step(super().prediction_step, model, inputs, prediction_loss_only, ignore_keys)
        else:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

def set_wandb_project(project_name="GraphLLM"):
    import os
    os.environ["WANDB_PROJECT"] = project_name
    os.environ["WANDB_WATCH"] = "false"


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    from .text_graph_dataset import TextGraphDataset, generate_text_graph_example, prepare_example_labels
    from .text_graph_collator import GraphCollator

    # create dataset and collator instances
    ds = TextGraphDataset.load("./src/experiments/expressiveness/1k_dataset.gtds")
    collator = GraphCollator(tokenizer=tokenizer)

    # load the model
    from ..models.llama_attn_bias import GraphLlamaForCausalLM
    model = GraphLlamaForCausalLM.from_pretrained(model_name)

    # ---------------------------------------------------------------
    # prepare training arguments and trainer
    from transformers import TrainingArguments
    
    training_args = TrainingArguments(
        output_dir="./graph_llama_checkpoints",
        report_to="wandb",
        run_name="graph-llama-v1",
        logging_steps=1,
        gradient_accumulation_steps=4,
        
        # custom schedule via parameters
        optim="adamw_torch",
        learning_rate=5e-6,
        lr_scheduler_type="cosine_with_restarts",
        warmup_ratio=0.1,
        lr_scheduler_kwargs={"num_cycles": 1},
        
        per_device_train_batch_size=8,
        num_train_epochs=1,
        gradient_checkpointing=False, 
    )

    # initialize using the custom class
    trainer = GraphTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
    )

    trainer.train()