import torch
from einops import repeat, rearrange


class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = torch.nn.LayerNorm(dim)
        self.norm_context = (
            torch.nn.LayerNorm(context_dim) if exists(context_dim) else None
        )

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)
        return self.fn(x, **kwargs)


class GEGLU(torch.nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gates)


class FeedForward(torch.nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            torch.nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class Attention(torch.nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = torch.nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = torch.nn.Linear(inner_dim, query_dim, bias=False)

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_mem_efficient=True
        ):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class LatentAttentionModel(torch.nn.Module):
    # config_class = LatentAttentionConfig

    def __init__(self):
        super().__init__()
        self.in_layer = torch.nn.Linear(4096, 512)
        ## cross-attention block
        # num_latents, latent_dim, cross_heads, cross_dim_head = 512, 4096, 8, 4096
        num_latents, latent_dim, cross_heads, cross_dim_head = 32, 512, 2, 64
        # num_latents, latent_dim, cross_heads, cross_dim_head = (
        #     config.num_latents_value,
        #     config.latent_dim,
        #     config.num_cross_heads,
        #     config.cross_dim_head,
        # )
        # dim = config.hidden_dim
        # dim = 4096
        dim = 512
        # init latent_attention and latents
        self.cross_attend_blocks = torch.nn.ModuleList(
            [
                PreNorm(
                    latent_dim,
                    Attention(
                        latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head
                    ),
                    context_dim=dim,
                ),
                PreNorm(latent_dim, FeedForward(latent_dim)),
            ]
        )
        # self.output_normalize = config.output_normalize
        self.output_normalize = True
        self.register_parameter(
            "latents", torch.nn.Parameter(torch.randn(num_latents, latent_dim))
        )
        self.out_layer = torch.nn.Linear(512, 4096)

    def forward(self, embeddings, attention_mask: torch.Tensor = None):
        ## cross-attention block
        hiddens = embeddings
        cross_attn, cross_ff = self.cross_attend_blocks
        b, *_, device = *hiddens.shape, hiddens.device
        hiddens = self.in_layer(hiddens)
        x = repeat(self.latents, "n d -> b n d", b=b)
        hiddens = cross_attn(hiddens, context=x, mask=None) + hiddens
        hiddens = cross_ff(hiddens) + hiddens
        hiddens = self.out_layer(hiddens)
        if attention_mask != None:
            s = torch.sum(hiddens * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            hiddens = s / d
            if self.output_normalize:
                hiddens = torch.nn.functional.normalize(hiddens, p=2, dim=-1)
        return hiddens
