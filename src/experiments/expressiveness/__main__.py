"""
Expressiveness experiment — refactor-validation harness.

Runs the HARD connectivity task across the v0 and v2 GTLM implementations on a
single shared `.gtds` dataset (the `TextGraphDataset` is implementation-agnostic;
only the collator / forward contract / trainer differ), so we can confirm:

  (a) the refactored v2 model still learns the task — accuracy parity with v0;
  (b) it is faster / leaner at scale — throughput + peak memory on large graphs.

Selectable implementations (``IMPL``):
    v0-eager   GraphLlamaForCausalLM (legacy)  + GraphCollator    + GraphTrainer
    v2-eager   GTLMLlamaForCausalLM            + GraphCollatorV2   + GraphTrainerV2
    v2-flex        "             (pad_to_block) +     "            +     "

Modes (``MODE``):
    train  small graphs, train to convergence  -> accuracy / parity
    bench  large graphs, few-step throughput + peak CUDA memory
"""

from ...utils import (
    set_wandb_project,
    GraphTrainer, GraphTrainerV2,
    TextGraphDataset, GraphCollator, GraphCollatorV2,
    make_compute_metrics, shift_logits_for_metrics,
)
from ...models.legacy.modeling_gtlm_llama_v0 import GraphLlamaForCausalLM, GraphLlamaConfig
from ...models import GTLMLlamaForCausalLM, GTLMLlamaConfig

from .data_gen import create_and_save_dataset, dataset_path_and_size, prepare_dataset

import os
import time
import statistics
import torch
from transformers import TrainingArguments, AutoTokenizer, set_seed
from peft import LoraConfig, get_peft_model
from accelerate.utils import send_to_device

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Which eval metric each implementation's trainer reports as its accuracy.
ACCURACY_METRIC = {"v0": "eval_classification_accuracy", "v2": "eval_em_accuracy"}

# Parameter-name substrings of the trainable graph-bias parameters (everything
# else stays frozen). The two implementations name these modules differently:
#   v0  ...self_attn.{spd_weights, rrwp_proj, magnetic_*}
#   v2  ...self_attn.graph_bias.bias_modules.* (one umbrella prefix)
ACTIVE_PARAMS_V0 = ["spd_weights", "laplacian_weights", "rwse_weights", "rrwp_proj", "magnetic_"]
ACTIVE_PARAMS_V2 = ["graph_bias"]

def active_params_for(impl):
    """Return the trainable-bias name substrings for ``impl``'s parameter layout."""
    version, _ = parse_impl(impl)
    return ACTIVE_PARAMS_V0 if version == "v0" else ACTIVE_PARAMS_V2

#region -------- Implementation dispatch (v0 vs v2) --------
def parse_impl(impl):
    """Split an IMPL string like 'v2-flex' into ('v2', 'flex') with validation."""
    version, _, backend = impl.partition("-")
    if version not in ("v0", "v2"):
        raise ValueError(f"Unknown version '{version}' in IMPL='{impl}' (expected v0|v2).")
    if backend not in ("eager", "flex"):
        raise ValueError(f"Unknown backend '{backend}' in IMPL='{impl}' (expected eager|flex).")
    if version == "v0" and backend != "eager":
        raise ValueError(f"v0 only supports the eager backend; got IMPL='{impl}'.")
    return version, backend

def build_model(impl, model_name, bias_params, k_hop, k_hop_directed, device, flex_compile_mode=None):
    """Construct the model for ``impl`` and freeze every parameter.

    v0 is dense and eager-only (its config has no ``k_hop`` / backend fields); the
    ``k_hop`` / backend knobs are v2 concepts and are simply not passed to v0.
    """
    version, backend = parse_impl(impl)
    if version == "v0":
        config = GraphLlamaConfig.from_pretrained(model_name, **bias_params)
        model = GraphLlamaForCausalLM.from_pretrained(model_name, config=config, attn_implementation="eager")
    else:
        extra = dict(k_hop=k_hop, k_hop_directed=k_hop_directed, graph_attn_impl=backend)
        if flex_compile_mode is not None:
            extra["flex_compile_mode"] = flex_compile_mode
        config = GTLMLlamaConfig.from_pretrained(model_name, **bias_params, **extra)
        model = GTLMLlamaForCausalLM.from_pretrained(model_name, config=config, graph_attn_impl=backend)

    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def build_collator(impl, tokenizer, pad_token_id, bias_params, k_hop, k_hop_directed):
    """Build the collator matching ``impl`` (v0 ragged batch vs v2 packed batch)."""
    version, backend = parse_impl(impl)
    if version == "v0":
        return GraphCollator()
    magnetic_m = bias_params["magnetic_dim"] if bias_params.get("magnetic") else 0
    return GraphCollatorV2(
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        k_hop=k_hop,
        k_hop_directed=k_hop_directed,
        magnetic_m=magnetic_m,
        pad_to_block=(backend == "flex"),   # flex needs block-aligned L and bucketed N
    )

def forward_loss(impl, model, batch, device):
    """Run a single forward and return the loss, using the right calling convention."""
    version, _ = parse_impl(impl)
    batch = send_to_device(batch, device)
    if version == "v0":
        labels = batch.pop("labels", None)
        if labels is None:
            raise ValueError("Collated batch is missing 'labels'.")
        outputs = model(input_ids=None, input_graph_batch=batch, labels=labels)
    else:
        outputs = model(**batch)
    return outputs.loss
#endregion

#region -------- Parameter selection --------
def select_active_params(model, active_params=None, lora=None):
    """
    Applies LoRA if a configuration dictionary is provided.
    Sets requires_grad=True for parameters whose names contain any of the substrings in active_params.
    If active_params is None, no additional parameters are unfrozen.
    If active_params is "all", all parameters are set to requires_grad=True.
    """
    if lora is not None:
        print("Applying LoRA with config:", lora)
        lora_config = LoraConfig(
            r=lora.get("r", 8),
            lora_alpha=lora.get("lora_alpha", 16),
            target_modules=lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            lora_dropout=lora.get("lora_dropout", 0.05),
            bias=lora.get("bias", "none"),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    if active_params == "all":
        for param in model.parameters():
            param.requires_grad = True
    elif active_params is not None:
        for name, param in model.named_parameters():
            if any(active_param in name for active_param in active_params):
                param.requires_grad = True

    return model

def print_trainable_parameters(model):
    """Prints the number of trainable parameters, splitting LoRA vs custom bias params."""
    trainable_lora_params = 0
    trainable_custom_params = 0
    all_param = 0

    print("List of custom active parameters (requires_grad=True):")
    for name, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            if "lora" in name.lower():
                trainable_lora_params += num_params
                print(f" - {name} ({num_params:,} params) [LoRA Adapter]")
            else:
                trainable_custom_params += num_params
                print(f" - {name} ({num_params:,} params)")

    total_trainable = trainable_lora_params + trainable_custom_params
    print("\n" + "=" * 50)
    print("TRAINABLE PARAMETER SUMMARY")
    print("=" * 50)
    print(f"LoRA Adapters:       {trainable_lora_params:>15,}")
    print(f"Custom Graph Biases: {trainable_custom_params:>15,}")
    print("-" * 50)
    print(f"Total Trainable:     {total_trainable:>15,}")
    print(f"Total Model Params:  {all_param:>15,}")
    print(f"Trainable %:         {100 * total_trainable / all_param:>14.4f}%")
    print("=" * 50 + "\n")
#endregion

#region -------- Custom evaluation (v0 only; v2 uses the trainer's own helpers) --------
def smuggle_prediction_step(super_prediction_step, model, inputs, prediction_loss_only, ignore_keys=None):
    # Run the standard evaluation step
    loss, logits, labels = super_prediction_step(
        model, inputs, prediction_loss_only, ignore_keys
    )

    # SMUGGLE THE DATA (to the preprocess_logits_for_metrics function)
    batch_size = len(inputs["input_ids"])
    prediction_token_indices = torch.full((batch_size,), -1, device=logits.device)
    if labels is not None and not prediction_loss_only:
        for i in range(batch_size):
            example_input_len = sum([input_ids.shape[0] for input_ids in inputs["input_ids"][i]])
            prediction_token_indices[i] = example_input_len - 2  # the second-to-last token is the prediction site

        # Turn labels into a tuple: (actual_labels, prediction_token_indices)
        labels = (labels, prediction_token_indices)

    return loss, logits, labels

class PreprocessLogits:
    def __init__(self, label_options, pad_token_id):
        self.label_options = label_options
        self.pad_token_id = pad_token_id

    def __call__(self, logits, labels):
        labels, prediction_token_indices = labels  # unpack the tuple created in the smuggled prediction step

        batch_size = logits.shape[0]
        probs = torch.zeros((batch_size, 3), device=logits.device)  # [label_1_prob, label_2_prob, label_id]

        probability_distributions = torch.softmax(logits, dim=-1)
        for i in range(batch_size):
            prediction_token_index = prediction_token_indices[i]
            if prediction_token_index == -1:
                raise ValueError("Prediction token index not found in labels.")
            for j, label_option in enumerate(self.label_options):
                probs[i, j] = probability_distributions[i, prediction_token_index, label_option[0]]
            probs[i, 2] = 0 if labels[i][-1].item() == self.label_options[0][0] else (1 if labels[i][-1].item() == self.label_options[1][0] else -1)

        return probs

class ComputeMetrics:
    def __init__(self, label_options):
        self.label_options = label_options

    def __call__(self, eval_preds):
        probs, (labels, prediction_token_indices) = eval_preds

        total = 0.0
        positive_prediction_count = 0.0
        correct = 0.0
        for i in range(probs.shape[0]):
            if probs[i, 2] == -1:
                raise ValueError("True label not found in labels.")
            if probs[i, 2] == 0 and probs[i, 0] > probs[i, 1]:
                correct += 1
            elif probs[i, 2] == 1 and probs[i, 1] > probs[i, 0]:
                correct += 1
            if probs[i, 0] > probs[i, 1]:
                positive_prediction_count += 1
            total += 1

        return {
            "classification_accuracy": float(correct) / float(total) if total > 0 else 0.0,
            "positive_prediction_ratio": float(positive_prediction_count) / float(total) if total > 0 else 0.0,
        }
#endregion

#region -------- Training --------
def make_trainer(impl, model, training_args, train_dataset, eval_dataset, collator,
                 active_params, bias_lr, label_options, pad_token_id):
    """Build the right Trainer + metrics for ``impl``.

    v0 needs the prediction-step surgery (`smuggle_prediction_step`) to score the
    Yes/No token; v2 emits standard (B, L) labels so the trainer's own
    `shift_logits_for_metrics` + `make_compute_metrics` (prompt-only exact match)
    suffice.
    """
    version, _ = parse_impl(impl)
    if version == "v0":
        if label_options is None or pad_token_id is None:
            raise ValueError("v0 evaluation needs label_options and pad_token_id.")
        return GraphTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            compute_metrics=ComputeMetrics(label_options=label_options),
            preprocess_logits_for_metrics=PreprocessLogits(label_options=label_options, pad_token_id=pad_token_id),
            custom_prediction_step=smuggle_prediction_step,
            active_params=active_params,
            bias_lr=bias_lr,
        )
    return GraphTrainerV2(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=make_compute_metrics(),
        preprocess_logits_for_metrics=shift_logits_for_metrics,
        active_params=active_params,
        bias_lr=bias_lr,
    )

def training_run(
    impl,
    model,
    train_dataset,
    eval_dataset,
    collator,
    run_name,
    num_epochs=3,
    batch_size=8,
    learning_rate=5e-5,
    bias_learning_rate=1e-3,
    accumulation_steps=4,
    label_options=None,
    pad_token_id=None,
    active_params=None,
    report_to="none",
    eval_steps=25,
    seed=42,
):
    version, _ = parse_impl(impl)
    metric = ACCURACY_METRIC[version]
    steps_per_epoch = max(1, len(train_dataset) // batch_size // accumulation_steps)

    training_args = TrainingArguments(
        num_train_epochs=num_epochs,
        output_dir=f"./checkpoints/{run_name}",
        seed=seed,
        data_seed=seed,
        logging_steps=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        gradient_checkpointing=False,

        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        metric_for_best_model=metric,
        greater_is_better=True,
        save_total_limit=2,
        load_best_model_at_end=True,

        report_to=report_to,
        run_name=run_name,

        learning_rate=learning_rate,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": learning_rate / 10},
        warmup_steps=steps_per_epoch,
        weight_decay=0.1,
    )

    trainer = make_trainer(
        impl, model, training_args, train_dataset, eval_dataset, collator,
        active_params=active_params, bias_lr=bias_learning_rate,
        label_options=label_options, pad_token_id=pad_token_id,
    )

    trainer.train()
    metrics = trainer.evaluate()
    return metrics.get(metric)
#endregion

#region -------- Benchmark (large graphs: throughput + peak memory) --------
def bench_impl(impl, examples, bias_params, model_name, k_hop, k_hop_directed,
               batch_size, num_warmup, num_iters, device, flex_compile_mode):
    """Time a few train steps for one impl on a fixed pool of large graphs.

    Returns (ms_per_step, peak_mem_gb) or (None, None) on CUDA OOM.
    """
    model, tokenizer = build_model(
        impl, model_name, bias_params, k_hop, k_hop_directed, device, flex_compile_mode
    )
    model = select_active_params(model, active_params=active_params_for(impl))
    model.train()
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collator = build_collator(impl, tokenizer, pad_token_id, bias_params, k_hop, k_hop_directed)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    def make_batch(step):
        lo = (step * batch_size) % max(1, len(examples))
        chunk = examples[lo:lo + batch_size]
        if len(chunk) < batch_size:
            chunk = examples[:batch_size]
        return collator(chunk)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    try:
        # Warmup (absorbs flex's one-time torch.compile on the first step).
        for s in range(num_warmup):
            loss = forward_loss(impl, model, make_batch(s), device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for s in range(num_iters):
            loss = forward_loss(impl, model, make_batch(num_warmup + s), device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ms_per_step = (time.perf_counter() - t0) / num_iters * 1000.0
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else float("nan")
        return ms_per_step, peak_gb
    except torch.cuda.OutOfMemoryError:
        return None, None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return None, None
        raise
    finally:
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

def run_bench(impls, sizes, bias_params, model_name, k_hop, k_hop_directed,
              batch_size, num_warmup, num_iters, num_examples, spectral_dims,
              device, flex_compile_mode):
    """Generate small in-memory HARD datasets at each size and benchmark every impl."""
    rows = []
    for size in sizes:
        print(f"\n[bench] generating {num_examples} HARD graphs of size {size}...")
        ds = prepare_dataset(
            num_examples,
            min_size=size, max_size=size,
            spectral_dims=spectral_dims,
            tokenizer_name=model_name,
            max_rwse_steps=bias_params["max_rw_steps"],
            max_rrwp_steps=bias_params["max_rw_steps"],
            easy=False,
            magnetic_q=bias_params["magnetic_q"],
        )
        examples = [ds[i] for i in range(len(ds))]
        for impl in impls:
            print(f"[bench] size={size} impl={impl} k_hop={k_hop} ...")
            ms, peak = bench_impl(
                impl, examples, bias_params, model_name, k_hop, k_hop_directed,
                batch_size, num_warmup, num_iters, device, flex_compile_mode,
            )
            status = "OK" if ms is not None else "OOM"
            rows.append((size, impl, k_hop, ms, peak, status))

    print("\n" + "=" * 72)
    print(f"{'size':>6} | {'impl':<9} | {'k_hop':>5} | {'ms/step':>9} | {'peak GB':>8} | status")
    print("-" * 72)
    for size, impl, kh, ms, peak, status in rows:
        ms_str = f"{ms:9.1f}" if ms is not None else f"{'—':>9}"
        peak_str = f"{peak:8.2f}" if peak is not None else f"{'—':>8}"
        print(f"{size:>6} | {impl:<9} | {kh:>5} | {ms_str} | {peak_str} | {status}")
    print("=" * 72)
    return rows
#endregion

def calculate_label_distribution(dataset, tokenized_labels):
    yes_count = 0
    no_count = 0
    for example in dataset:
        label = example["labels"][-1].item()  # last label token is the Yes/No answer
        if label == tokenized_labels[0][0]:
            yes_count += 1
        elif label == tokenized_labels[1][0]:
            no_count += 1
    total = yes_count + no_count
    yes_percentage = (yes_count / total) * 100 if total > 0 else 0
    no_percentage = (no_count / total) * 100 if total > 0 else 0
    return yes_percentage, no_percentage

def load_or_create_dataset(dataset_size, easy, bias_params, model_name, min_nodes, max_nodes, spectral_dims):
    """Load a `.gtds` dataset, generating it (with the scalable label scheme) if absent."""
    path, _ = dataset_path_and_size(dataset_size, easy=easy, min_nodes=min_nodes, max_nodes=max_nodes)
    if not os.path.exists(path):
        print(f"Dataset not found at {path}. Creating new dataset...")
        create_and_save_dataset(
            dataset_size=dataset_size, min_nodes=min_nodes, max_nodes=max_nodes,
            spectral_dims=spectral_dims, model_name=model_name,
            max_rrwp_steps=bias_params["max_rw_steps"], max_rwse_steps=bias_params["max_rw_steps"],
            easy=easy, magnetic_q=bias_params["magnetic_q"],
        )
    ds = TextGraphDataset.load(path)
    print(f"Loaded dataset from {path} with {len(ds)} examples.")
    return ds


if __name__ == "__main__":
    # ── Shared bias configuration (identical across v0 and v2) ──────────────────
    BIAS_PARAMS = {
        "spd": True,
        "max_spd": 8,
        "laplacian": False,
        "rwse": False,
        "rrwp": True,
        "max_rw_steps": 16,
        "magnetic": True,
        "magnetic_dim": 32,
        "magnetic_q": 0.25,
    }
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    SPECTRAL_DIMS = 16

    # ── What to run ─────────────────────────────────────────────────────────────
    MODE = "train"                              # "train" | "bench"
    IMPLS = ["v0-eager", "v2-eager"]  # comparison matrix (eager-only: v0↔v2 equivalence + k-hop effect)
    K_HOPS = [0, 1]                             # structural sparsity sweep (free param)
    K_HOP_DIRECTED = False                      # HARD graphs are bidirectional after to_directed()
    DIFFICULTY = "HARD"                         # this harness validates the HARD variant
    IS_EASY = DIFFICULTY == "EASY"
    REPORT_TO = "none"                          # "none" | "wandb"
    FLEX_COMPILE_MODE = "default"               # "default" (fast compile) | "max-autotune-no-cudagraphs"

    device = get_device()
    if REPORT_TO == "wandb":
        set_wandb_project("GraphLLM")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    possible_labels = [" Yes", " No"]
    tokenized_possible_labels = [tokenizer(label, add_special_tokens=False).input_ids for label in possible_labels]
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    run_suffix = "+".join(
        bias_type
        for bias_type in [
            f"spd({BIAS_PARAMS['max_spd']})", "laplacian", "rwse",
            f"rrwp({BIAS_PARAMS['max_rw_steps']})",
            f"magnetic(dim={BIAS_PARAMS['magnetic_dim']},q={BIAS_PARAMS['magnetic_q']})",
        ]
        if BIAS_PARAMS[bias_type.split('(')[0]]
    )

    if MODE == "train":
        # ── Small graphs: train each impl to convergence and report accuracy ────
        TRAIN_DATASET_SIZE = 2_000        # "light first" parity pass; raise for the full run
        EVAL_DATASET_SIZE = 500
        MIN_NODES, MAX_NODES = 10, 25
        LR = 4e-6
        BIAS_LR = 1e-2
        # Multiple training seeds → mean ± std per config, so the v0↔v2 equivalence
        # and the k_hop effect can be read against the run-to-run spread instead of
        # a single noisy point. Override with e.g. SEEDS="1,2" to run a subset.
        SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",") if s.strip()]

        train_dataset = load_or_create_dataset(TRAIN_DATASET_SIZE, IS_EASY, BIAS_PARAMS, MODEL_NAME, MIN_NODES, MAX_NODES, SPECTRAL_DIMS)
        eval_dataset = load_or_create_dataset(EVAL_DATASET_SIZE, IS_EASY, BIAS_PARAMS, MODEL_NAME, MIN_NODES, MAX_NODES, SPECTRAL_DIMS)

        train_yes, train_no = calculate_label_distribution(train_dataset, tokenized_possible_labels)
        eval_yes, eval_no = calculate_label_distribution(eval_dataset, tokenized_possible_labels)
        print("!" * 100)
        print(f"Training dataset label distribution: {train_yes:.2f}% Yes, {train_no:.2f}% No")
        print(f"Evaluation dataset label distribution: {eval_yes:.2f}% Yes, {eval_no:.2f}% No")
        print("!" * 100)

        results = {}  # (impl, k_hop, seed) -> accuracy
        for seed in SEEDS:
            for k_hop in K_HOPS:
                for impl in IMPLS:
                    version, _ = parse_impl(impl)
                    # v0 is dense (k_hop-agnostic); run it once at k_hop=0 only.
                    if version == "v0" and k_hop != K_HOPS[0]:
                        continue

                    # Reset RNG before each build so bias-param init + data order are
                    # fully determined by `seed`; this also init-pairs k0/k1 within a
                    # seed, isolating the k_hop effect from initialization noise.
                    set_seed(seed)

                    run_name = f"{DIFFICULTY}_{impl}_k{k_hop}_s{seed}_{run_suffix}"
                    print("\n" + "#" * 100)
                    print(f"# Training {impl} (k_hop={k_hop}, seed={seed}) — run '{run_name}'")
                    print("#" * 100)

                    model, _ = build_model(impl, MODEL_NAME, BIAS_PARAMS, k_hop, K_HOP_DIRECTED, device, FLEX_COMPILE_MODE)
                    active_params = active_params_for(impl)
                    model = select_active_params(model, active_params=active_params, lora=None)
                    print_trainable_parameters(model)

                    accuracy = training_run(
                        impl=impl,
                        model=model,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        collator=build_collator(impl, tokenizer, pad_token_id, BIAS_PARAMS, k_hop, K_HOP_DIRECTED),
                        run_name=run_name,
                        num_epochs=3,
                        batch_size=4,
                        learning_rate=LR,
                        bias_learning_rate=BIAS_LR,
                        accumulation_steps=8,
                        label_options=tokenized_possible_labels,
                        pad_token_id=pad_token_id,
                        active_params=active_params,
                        report_to=REPORT_TO,
                        seed=seed,
                    )
                    results[(impl, k_hop, seed)] = accuracy
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

        # Distinct (impl, k_hop) configs, in a stable order.
        configs = []
        for k_hop in K_HOPS:
            for impl in IMPLS:
                version, _ = parse_impl(impl)
                if version == "v0" and k_hop != K_HOPS[0]:
                    continue
                configs.append((impl, k_hop))

        print("\n" + "=" * 72)
        print(f"FINAL EVAL ACCURACY — mean ± std over seeds {SEEDS}")
        print("=" * 72)
        for (impl, k_hop) in configs:
            accs = [results[(impl, k_hop, s)] for s in SEEDS
                    if results.get((impl, k_hop, s)) is not None]
            if not accs:
                continue
            mean = sum(accs) / len(accs)
            std = statistics.stdev(accs) if len(accs) > 1 else 0.0
            accs_str = ", ".join(f"{a:.4f}" for a in accs)
            print(f"  {impl:<10} k_hop={k_hop}: {mean:.4f} ± {std:.4f}   (seeds: {accs_str})")
        print("=" * 72)

    elif MODE == "bench":
        # ── Large graphs: measure step-time and peak memory ─────────────────────
        SIZES = [500, 1000]      # node counts to sweep
        BENCH_BATCH_SIZE = 2
        NUM_WARMUP = 2
        NUM_ITERS = 5
        NUM_EXAMPLES = 16

        for k_hop in K_HOPS:
            print("\n" + "#" * 72)
            print(f"# BENCH — k_hop={k_hop}")
            print("#" * 72)
            run_bench(
                impls=IMPLS,
                sizes=SIZES,
                bias_params=BIAS_PARAMS,
                model_name=MODEL_NAME,
                k_hop=k_hop,
                k_hop_directed=K_HOP_DIRECTED,
                batch_size=BENCH_BATCH_SIZE,
                num_warmup=NUM_WARMUP,
                num_iters=NUM_ITERS,
                num_examples=NUM_EXAMPLES,
                spectral_dims=SPECTRAL_DIMS,
                device=device,
                flex_compile_mode=FLEX_COMPILE_MODE,
            )

    else:
        raise ValueError(f"Unknown MODE='{MODE}' (expected 'train' or 'bench').")
