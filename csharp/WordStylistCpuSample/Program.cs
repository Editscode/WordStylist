using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace WordStylistCpuSample;

public static class Program
{
    private const string Alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    private const int PadToken = 52;
    private const int NumTokens = 1;

    public static int Main(string[] args)
    {
        var options = ParseArgs(args);
        if (!options.IsValid)
        {
            Console.Error.WriteLine("Usage: --unet <path> --vae <path> --output <png> --word <text> [--style <int>] [--steps <int>] [--seed <int>]");
            return 1;
        }

        var wordTokens = EncodeWord(options.Word, options.MaxSeqLen);
        var latents = RunDiffusion(options, wordTokens);
        SaveImage(options, latents);

        Console.WriteLine($"Saved image to {options.OutputPath}");
        return 0;
    }

    private static Options ParseArgs(string[] args)
    {
        var options = new Options
        {
            Steps = 1000,
            Style = 0,
            Seed = 0,
            ImgHeight = 64,
            ImgWidth = 256,
            MaxSeqLen = 10,
        };

        for (var i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--unet":
                    options.UnetPath = args[++i];
                    break;
                case "--vae":
                    options.VaePath = args[++i];
                    break;
                case "--output":
                    options.OutputPath = args[++i];
                    break;
                case "--word":
                    options.Word = args[++i];
                    break;
                case "--style":
                    options.Style = int.Parse(args[++i]);
                    break;
                case "--steps":
                    options.Steps = int.Parse(args[++i]);
                    break;
                case "--seed":
                    options.Seed = int.Parse(args[++i]);
                    break;
                case "--img-height":
                    options.ImgHeight = int.Parse(args[++i]);
                    break;
                case "--img-width":
                    options.ImgWidth = int.Parse(args[++i]);
                    break;
                case "--max-seq-len":
                    options.MaxSeqLen = int.Parse(args[++i]);
                    break;
            }
        }

        options.IsValid = !string.IsNullOrWhiteSpace(options.UnetPath)
                          && !string.IsNullOrWhiteSpace(options.VaePath)
                          && !string.IsNullOrWhiteSpace(options.OutputPath)
                          && !string.IsNullOrWhiteSpace(options.Word);
        return options;
    }

    private static int[] EncodeWord(string word, int maxSeqLen)
    {
        var tokens = new List<int>(maxSeqLen);
        foreach (var ch in word)
        {
            var index = Alphabet.IndexOf(ch);
            if (index < 0)
            {
                throw new ArgumentException($"Unsupported character '{ch}'. Only [A-Za-z] are allowed.");
            }

            tokens.Add(index + NumTokens);
        }

        while (tokens.Count < maxSeqLen)
        {
            tokens.Add(PadToken);
        }

        if (tokens.Count > maxSeqLen)
        {
            tokens.RemoveRange(maxSeqLen, tokens.Count - maxSeqLen);
        }

        return tokens.ToArray();
    }

    private static float[] RunDiffusion(Options options, int[] tokens)
    {
        var random = new Random(options.Seed);
        var latentH = options.ImgHeight / 8;
        var latentW = options.ImgWidth / 8;
        var latentSize = 4 * latentH * latentW;
        var x = new float[latentSize];
        for (var i = 0; i < x.Length; i++)
        {
            x[i] = NextGaussian(random);
        }

        var (betas, alphas, alphaHats) = BuildNoiseSchedule(options.Steps);

        using var session = new InferenceSession(options.UnetPath);
        var contextTensor = new DenseTensor<long>(new[] { 1, options.MaxSeqLen });
        for (var i = 0; i < options.MaxSeqLen; i++)
        {
            contextTensor[0, i] = tokens[i];
        }

        var labelTensor = new DenseTensor<long>(new[] { 1 });
        labelTensor[0] = options.Style;

        for (var step = options.Steps - 1; step >= 1; step--)
        {
            var inputTensor = new DenseTensor<float>(x, new[] { 1, 4, latentH, latentW });
            var tTensor = new DenseTensor<long>(new[] { 1 });
            tTensor[0] = step;

            using var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("x", inputTensor),
                NamedOnnxValue.CreateFromTensor("timesteps", tTensor),
                NamedOnnxValue.CreateFromTensor("context", contextTensor),
                NamedOnnxValue.CreateFromTensor("y", labelTensor),
            };

            using var results = session.Run(inputs);
            var predictedNoise = results[0].AsTensor<float>();

            var alpha = alphas[step];
            var alphaHat = alphaHats[step];
            var beta = betas[step];
            var sqrtAlpha = MathF.Sqrt(alpha);
            var sqrtOneMinusAlphaHat = MathF.Sqrt(1f - alphaHat);
            var sqrtBeta = MathF.Sqrt(beta);

            for (var i = 0; i < x.Length; i++)
            {
                var noise = step > 1 ? NextGaussian(random) : 0f;
                var pred = predictedNoise.Buffer.Span[i];
                x[i] = (1f / sqrtAlpha) * (x[i] - ((1f - alpha) / sqrtOneMinusAlphaHat) * pred) + sqrtBeta * noise;
            }
        }

        for (var i = 0; i < x.Length; i++)
        {
            x[i] = x[i] / 0.18215f;
        }

        return x;
    }

    private static void SaveImage(Options options, float[] latents)
    {
        var latentH = options.ImgHeight / 8;
        var latentW = options.ImgWidth / 8;
        using var session = new InferenceSession(options.VaePath);
        var latentTensor = new DenseTensor<float>(latents, new[] { 1, 4, latentH, latentW });

        using var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("latents", latentTensor),
        };

        using var results = session.Run(inputs);
        var images = results[0].AsTensor<float>();
        var image = new Image<Rgb24>(options.ImgWidth, options.ImgHeight);

        for (var y = 0; y < options.ImgHeight; y++)
        {
            for (var x = 0; x < options.ImgWidth; x++)
            {
                var r = ClampToByte(images[0, 0, y, x]);
                var g = ClampToByte(images[0, 1, y, x]);
                var b = ClampToByte(images[0, 2, y, x]);
                image[x, y] = new Rgb24(r, g, b);
            }
        }

        Directory.CreateDirectory(Path.GetDirectoryName(options.OutputPath) ?? ".");
        image.Save(options.OutputPath);
    }

    private static byte ClampToByte(float value)
    {
        var clamped = Math.Clamp(value, 0f, 1f);
        return (byte)Math.Round(clamped * 255f);
    }

    private static (float[] Betas, float[] Alphas, float[] AlphaHats) BuildNoiseSchedule(int steps)
    {
        var betas = new float[steps];
        var alphas = new float[steps];
        var alphaHats = new float[steps];
        const float betaStart = 1e-4f;
        const float betaEnd = 2e-2f;

        for (var i = 0; i < steps; i++)
        {
            betas[i] = betaStart + (betaEnd - betaStart) * i / (steps - 1f);
            alphas[i] = 1f - betas[i];
            alphaHats[i] = i == 0 ? alphas[i] : alphaHats[i - 1] * alphas[i];
        }

        return (betas, alphas, alphaHats);
    }

    private static float NextGaussian(Random random)
    {
        var u1 = 1.0 - random.NextDouble();
        var u2 = 1.0 - random.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }

    private sealed class Options
    {
        public string? UnetPath { get; set; }
        public string? VaePath { get; set; }
        public string? OutputPath { get; set; }
        public string? Word { get; set; }
        public int Style { get; set; }
        public int Steps { get; set; }
        public int Seed { get; set; }
        public int ImgHeight { get; set; }
        public int ImgWidth { get; set; }
        public int MaxSeqLen { get; set; }
        public bool IsValid { get; set; }
    }
}
