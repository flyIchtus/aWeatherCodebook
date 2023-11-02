import math
import random

import torch
from torch import nn

from  op import FusedLeakyReLU, conv2d_gradfix
from model.conditionalSockets import LinearSocket
import model.instance_discrim as ID

library = {'stylegan2' : {'G' :  'Generator', 'D' : 'Discriminator'}}

from model.stylegan_base_blocks import PixelNorm, EqualConv2d, EqualLinear, Upsample, Blur



class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        fused=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        self.fused = fused

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        if not self.fused:
            weight = self.scale * self.weight.squeeze(0)
            style = self.modulation(style)

            if self.demodulate:
                w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
                dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

            input = input * style.reshape(batch, in_channel, 1, 1)

            if self.upsample:
                weight = weight.transpose(0, 1)
                out = conv2d_gradfix.conv_transpose2d(
                    input, weight, padding=0, stride=2
                )
                out = self.blur(out)

            elif self.downsample:
                input = self.blur(input)
                out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)

            else:
                out = conv2d_gradfix.conv2d(input, weight, padding=self.padding)

            if self.demodulate:
                out = out * dcoefs.view(batch, -1, 1, 1)

            return out
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            #print("a4")
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            #print("a5")
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
            #print("a6")

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )
        #print("a7")

        if self.upsample:
            #print("a8")
            input = input.view(1, batch * in_channel, height, width)
            #print("a9")
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            #print("a10")
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            #print("a11")
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            #print("a12")
            _, _, height, width = out.shape
            #print("a13")
            out = out.view(batch, self.out_channel, height, width)
            #print("a14")
            out = self.blur(out)
            #print("a15")

        elif self.downsample:
            #print("a16")
            input = self.blur(input)
            #print("a17")
            _, _, height, width = input.shape
            #print("a18")
            input = input.view(1, batch * in_channel, height, width)
            #print("a19")
            out = conv2d_gradfix.conv2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            #print("a20")
            _, _, height, width = out.shape
            #print("a21")
            out = out.view(batch, self.out_channel, height, width)
            #print("a22")

        else:
            #print("a23")
            input = input.view(1, batch * in_channel, height, width)
            #print("a24")
            out = conv2d_gradfix.conv2d(
                input, weight, padding=self.padding, groups=batch
            )
            #print("a25")
            _, _, height, width = out.shape
            #print("a26")
            out = out.view(batch, self.out_channel, height, width)
            #print("a27")

        return out


class NoiseInjection(nn.Module):
    def __init__(self, use_noise=True):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))
        self.use_noise = use_noise

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise if self.use_noise else image


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        use_noise=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection(use_noise=use_noise)
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        #print("input styled ", input.shape)
        out = self.conv(input, style)
        #print("modulated styled ", out.shape)
        out = self.noise(out, noise=noise)
        #print("noise styled", out.shape)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1], nb_var=3):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, nb_var, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, nb_var, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias
        input_conved = torch.clone(out)
        

        if skip is not None:
            prev_rgb = torch.clone(skip)
            skip = self.upsample(skip)

            out = out + skip

        return (out, input_conved, skip, prev_rgb) if skip  is not None else (out, input_conved)


class condMapping(nn.Module):
    def __init__(self, style_dim, mlp_inj_level, lr_mlp, n_mlp, discrete_level=0, 
                 mid_ch=3, mid_h=32, mid_w=32,
                 ):
        super().__init__()
        layers = [PixelNorm()]
        self.mlp_inj_level = mlp_inj_level
        
        for i in range(mlp_inj_level):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )
        # injection of conditional input by concatenation on this level
        layers.append(
            EqualLinear(
               style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
            )
        )

        for i in range(mlp_inj_level, n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )


        self.style = nn.ModuleList(layers)
        self.Linear = LinearSocket(mid_ch + (discrete_level==1) , mid_h, mid_w, style_dim)
    
    def forward(self, s, y, lambda_t):

        y = self.Linear(y)
        for m in self.style[:self.mlp_inj_level]:
            s = m(s)
            
        o = s + lambda_t * y
        for m in self.style[self.mlp_inj_level:]:
            o = m(o)
        return o


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        mlp_inj_level,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        nb_var=3,
        use_noise=True,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim
        self.mlp_inj_level = mlp_inj_level

        self.style = condMapping(style_dim, mlp_inj_level, lr_mlp, n_mlp)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        self.chm = channel_multiplier
        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel, use_noise=use_noise
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False, nb_var=nb_var)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                    use_noise=use_noise
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel, use_noise=use_noise
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim, nb_var=nb_var))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        condition,
        lambda_t,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        return_rgb=False,
    ):
        if not input_is_latent:
           styles = [self.style(s,condition, lambda_t) for s in styles]


        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)
        #print("latent ", latent.shape)
        out = self.input(latent)
        #print("input gen ", out.shape)
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        #print("conv1 gen ", out.shape)

        skip, input_conved = self.to_rgb1(out, latent[:, 1])
        #print("rgb1 ", skip.shape)
        if return_rgb:
            rgbs_saved = {}
            rgbs_saved['prev_rgb'] = {}
            rgbs_saved['prev_rgb_upsampled'] = {}
            rgbs_saved['input_conved'] = {}
            rgbs_saved['current_rgb_out'] = {}
            
            rgbs_saved['prev_rgb'][1] = input_conved.detach().cpu().numpy()
            rgbs_saved['prev_rgb_upsampled'][1] = input_conved.detach().cpu().numpy()
            rgbs_saved['input_conved'][1] = input_conved.detach().cpu().numpy()
            rgbs_saved['current_rgb_out'][1] = skip.detach().cpu().numpy()

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            #print("conv1 gen ", out.shape)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            #print("conv2 gen ", out.shape)
            skip, input_conved, prev_rgb_upsampled, prev_rgb = to_rgb(out, latent[:, i + 2], skip)
            #print("rgb ", skip.shape)
            if return_rgb:
                rgbs_saved['prev_rgb'][i//2 + 2] = prev_rgb.detach().cpu().numpy()
                rgbs_saved['prev_rgb_upsampled'][i//2 + 2] = prev_rgb_upsampled.detach().cpu().numpy()
                rgbs_saved['input_conved'][i//2 + 2] = input_conved.detach().cpu().numpy()
                rgbs_saved['current_rgb_out'][i//2 + 2] = skip.detach().cpu().numpy()

            i += 2

        image = skip

        if return_latents:
            if return_rgb:
                return image, latent, rgbs_saved
            else:
                return image, latent, None

        else:
            if return_rgb:
                return image, None, rgbs_saved
            else:
                return image, None, None


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        #print("resblock, out1 ", out.shape)
        out = self.conv2(out)
        #print("resblock, out2 ", out.shape)
        skip = self.skip(input)
        #print("resblock, skip ", skip.shape)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], nb_var=3, discrete_level=0, mid_ch=3, mid_h=32, mid_w=32):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(nb_var, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_uc_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

        self.LinearEmbed = LinearSocket(mid_ch + (discrete_level==1), mid_h, mid_w, channels[4])
        self.final_c_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
        )

    def forward(self, input, condition, lambda_t, normalize=False, instance_disc=False, temperature=None):
        
        #print("input ", input.shape)
        out = self.convs(input)

        #print("before std ", out.shape)
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        
        #print('std ', out.shape)
        out = self.final_conv(out)
        #print('final_conv ',out.shape)
        out = out.view(batch, -1)
        #print('out ',out.shape, self.final_linear[0].weight.shape, self.final_linear[1].weight.shape)
        out_inp = self.final_c_linear(out)

        y = self.LinearEmbed(condition)
            
        if instance_disc:
            
            out_cond = self.convs(condition) 
            out_cond = self.final_conv(out_cond)
            #print('final_conv ',out.shape)
            out_cond = out_cond.view(batch, -1)
            out_cond = self.final_c_linear(out_cond)

            instance_reg = ID.tripletNTXent(y, out_inp, out_cond, temperature) # anchor, positive, negative could be triplet loss with temperature used as margin
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)

        out = torch.cat([out, stddev], 1)

        uc_out = self.final_uc_linear(out)
        
        #we want to align embeddings with condition
        
        if normalize:
            out_inp = torch.nn.CosineSimilarity(y, out_inp)
        else:
            out_inp = torch.sum(y * out_inp, dim=(1,),keepdims=True)
        
        res = uc_out + lambda_t * out_inp + instance_reg if instance_disc else uc_out + lambda_t * out_inp
        return res