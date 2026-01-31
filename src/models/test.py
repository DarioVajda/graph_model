from transformers import LlamaForCausalLM, LlamaConfig, LlamaModel


class LlamaGraphLanguageModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        
        # add custom modifications here

    def forward(self, *args, **kwargs):
        # call original forward method
        outputs = super().forward(*args, **kwargs)
        
        # add custom processing here if needed

        return outputs


if __name__ == "__main__":
    language_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

    #region INSPECT MODEL:
    print(language_model)

    # OUTPUT:
        # LlamaForCausalLM(
        #   (model): LlamaModel(
        #     (embed_tokens): Embedding(128256, 2048)
        #     (layers): ModuleList(
        #       (0-15): 16 x LlamaDecoderLayer(
        #         (self_attn): LlamaAttention(
        #           (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
        #           (k_proj): Linear(in_features=2048, out_features=512, bias=False)
        #           (v_proj): Linear(in_features=2048, out_features=512, bias=False)
        #           (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
        #         )
        #         (mlp): LlamaMLP(
        #           (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
        #           (up_proj): Linear(in_features=2048, out_features=8192, bias=False)
        #           (down_proj): Linear(in_features=8192, out_features=2048, bias=False)
        #           (act_fn): SiLU()
        #         )
        #         (input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
        #         (post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
        #       )
        #     )
        #     (norm): LlamaRMSNorm((2048,), eps=1e-05)
        #     (rotary_emb): LlamaRotaryEmbedding()
        #   )
        #   (lm_head): Linear(in_features=2048, out_features=128256, bias=False)
        # )

    #endregion

        
    # simple test to see how matrix multipliation works with batched inputs
    # 1st shape: [2, 4, 6] <=> [batch_size, seq_length, spectral_dims]
    A = torch.tensor([
        [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24]
        ],
        [
            [1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4]
        ]
    ], dtype=torch.float32)
    # 2nd shape: [2, 3, 6] <=> [batch_size, d_head//2, spectral_dims]
    B = torch.tensor([
        [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18]
        ],
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ]
    ], dtype=torch.float32).transpose(1, 2)[0,:,:]  # Transpose to get shape [2, 6, 3] <=> [batch_size, spectral_dims, d_head//2]

    # output shape: [2, 4, 3] <=> [batch_size, seq_length, spectral_dims]
    C = A@B
    print("Test matrix multiplication result C.shape:", C.shape)
    print("A\n",A)
    print("B\n",B)
    print("C\n",C)

    exit()