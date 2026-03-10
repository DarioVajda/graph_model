import os, json
import torch
from transformers import Trainer
from peft import PeftModel

from ..models.llama_utils import save_bias_parameters

class GraphTrainer(Trainer):
    def __init__(self, *args, custom_prediction_step=None, active_params=None, bias_lr=None, **kwargs):
        # 1. FORCE KEEP UNUSED COLUMNS
        # We intercept the TrainingArguments during initialization to ensure 
        # the Trainer never deletes our custom 'input_graph_batch' data.
        if "args" in kwargs and kwargs["args"] is not None:
            kwargs["args"].remove_unused_columns = False

        self.custom_prediction_step = custom_prediction_step
        self.active_params = active_params
        self.bias_lr = bias_lr
            
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

        # 1. Grab the labels before compute_loss pops them out of `inputs`!
        raw_labels = inputs.get("labels")
        
        # 2. Standard forward pass
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                
        if prediction_loss_only:
            return (loss, None, None)
            
        logits = outputs.logits  # Shape: (batch_size, max_total_seq_len, vocab_size)
        
        # 3. Re-calculate prefix and prompt lengths to know WHERE the prompt is
        prepared = model._prepare_inputs(inputs['input_ids'], inputs['prompt_node'], padding_side="right")
        prefix_lengths = prepared['prefix_lengths']
        prompt_lengths = prepared['prompt_lengths']
        
        # 4. Convert logits to predictions and shift them (logit at t predicts t+1)
        preds = torch.argmax(logits, dim=-1)
        shifted_preds = torch.full_like(preds, fill_value=-100)
        shifted_preds[:, 1:] = preds[:, :-1]
        
        # 5. Extract ONLY the prompt predictions and pad everything cleanly
        batch_size = preds.shape[0]
        max_prompt_len = max(prompt_lengths) if prompt_lengths else 0
        
        prompt_preds = torch.full((batch_size, max_prompt_len), fill_value=-100, device=preds.device)
        padded_labels = torch.full((batch_size, max_prompt_len), fill_value=-100, device=preds.device)
        
        for i in range(batch_size):
            start = prefix_lengths[i]
            end = start + prompt_lengths[i]
            
            # Slice the exact predictions for the prompt
            prompt_preds[i, :prompt_lengths[i]] = shifted_preds[i, start:end]
            
            # Pad the raw labels so HF Trainer can safely accumulate them
            if raw_labels is not None and i < len(raw_labels):
                padded_labels[i, :raw_labels[i].shape[0]] = raw_labels[i]
                
        return (loss, prompt_preds, padded_labels)

    def save_model(self, output_dir=None, _internal_call=False):
        """
        Overrides the deafult Trainer save behavour to ensure lightweight checkpoints for both LoRA and non-LoRA custom fine-tuning.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Check if the model is wrapped in PEFT/LoRA
        is_peft = False
        if PeftModel is not None:
            is_peft = isinstance(self.model, PeftModel)

        if is_peft:
            super().save_model(output_dir)
        
        # save a configuration file that points to the base model
        base_model_name = self.model.config._name_or_path
        bias_config_data = {
            "base_model_name_or_path": base_model_name
        }
        with open(os.path.join(output_dir, "graph_bias_config.json"), "w") as f:
            json.dump(bias_config_data, f, indent=4)
        
        # save the regular model config file to preserve the custom configuration of the bias parameteres
        config = self.model.config
        config_path = os.path.join(output_dir, "config.json")
        config.save_pretrained(output_dir)

        # In both cases: Extract and save the custom graph bias-related parameters
        save_bias_parameters(self.model, output_dir, params=self.active_params)

    def create_optimizer(self):
        """
        Setup the optimizer with custom bias learning rates, perfectly 
        mirroring Hugging Face's internal checks and fallbacks.
        """
        # 1. Handle potential SageMaker or distributed wrapping
        from transformers.utils import is_sagemaker_mp_enabled
        is_smp = is_sagemaker_mp_enabled()
        opt_model = self.model_wrapped if is_smp else self.model

        if self.optimizer is None:
            # 2. Use HF's native, safer method to detect weight decay parameters
            decay_parameters = self.get_decay_parameter_names(opt_model)
            
            # 3. Retrieve the correct optimizer class and arguments
            if getattr(self, "optimizer_cls_and_kwargs", None) is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # 4. Custom Parameter Grouping
            groups = {
                "base_decay": [], "base_no_decay": [],
                "bias_decay": [], "bias_no_decay": [],
            }
            
            active_p = self.active_params or []
            
            for n, p in opt_model.named_parameters():
                if not p.requires_grad:
                    continue
                
                is_active = any(act in n for act in active_p)
                has_decay = n in decay_parameters
                
                if is_active and has_decay: groups["bias_decay"].append(p)
                elif is_active and not has_decay: groups["bias_no_decay"].append(p)
                elif not is_active and has_decay: groups["base_decay"].append(p)
                else: groups["base_no_decay"].append(p)

            base_lr = self.args.learning_rate
            bias_lr = self.bias_lr if self.bias_lr is not None else base_lr

            optimizer_grouped_parameters = [
                {"params": groups["base_decay"], "weight_decay": self.args.weight_decay, "lr": base_lr, "is_bias": False},
                {"params": groups["base_no_decay"], "weight_decay": 0.0, "lr": base_lr, "is_bias": False},
                {"params": groups["bias_decay"], "weight_decay": self.args.weight_decay, "lr": bias_lr, "is_bias": True},
                {"params": groups["bias_no_decay"], "weight_decay": 0.0, "lr": bias_lr, "is_bias": True},
            ]
            
            # Filter out empty groups to prevent optimizer crashes
            optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g["params"]) > 0]

            # 5. Handle special HF optimizer overwrites (GaLore, LOMO, layer-wise dummy)
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            # 6. Adam8bit compatibility (if you use bitsandbytes)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                for module in opt_model.modules():
                    if isinstance(module, torch.nn.Embedding): # Note: assumes torch is imported
                        manager.register_module_override(module, "weight", {"optim_bits": 32})

        # 7. Wrap for SageMaker if needed
        if is_smp:
            import smdistributed.modelparallel.torch as smp
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def log(self, logs: dict, *args, **kwargs) -> None:
        """
        Intercepts the logging dictionary to add the bias learning rate 
        only if a custom bias_lr was actually provided.
        """
        # Inject our custom learning rate into the logs dict
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                if group.get("is_bias", False):
                    logs["bias_learning_rate"] = group["lr"]
                    break
        
        # Pass the augmented logs, plus any extra arguments HF throws at us, back to the default logger
        super().log(logs, *args, **kwargs)



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