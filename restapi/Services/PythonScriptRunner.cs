using System.Diagnostics;
using Microsoft.Extensions.Options;
using WordStylist.Api.Models;
using WordStylist.Api.Options;

namespace WordStylist.Api.Services;

public sealed class PythonScriptRunner : IPythonScriptRunner
{
    private const string SamplingScriptName = "sampling.py";

    private readonly ScriptRunnerOptions _options;
    private readonly ILogger<PythonScriptRunner> _logger;

    public PythonScriptRunner(IOptions<ScriptRunnerOptions> options, ILogger<PythonScriptRunner> logger)
    {
        _options = options.Value;
        _logger = logger;
    }

    public async Task<ScriptRunResult> RunSamplingAsync(GenerateRequest request, CancellationToken cancellationToken)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = _options.PythonPath,
            WorkingDirectory = _options.ScriptsRoot,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        startInfo.ArgumentList.Add(SamplingScriptName);
        startInfo.ArgumentList.Add("--prompt");
        startInfo.ArgumentList.Add(request.Prompt);
        startInfo.ArgumentList.Add("--steps");
        startInfo.ArgumentList.Add(request.Steps.ToString());

        if (request.Seed is not null)
        {
            startInfo.ArgumentList.Add("--seed");
            startInfo.ArgumentList.Add(request.Seed.Value.ToString());
        }

        using var process = new Process { StartInfo = startInfo };
        _logger.LogInformation("Starting python script {Script} with prompt length {PromptLength}",
            SamplingScriptName,
            request.Prompt.Length);

        process.Start();

        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeoutCts.CancelAfter(TimeSpan.FromSeconds(_options.TimeoutSeconds));

        var stdoutTask = process.StandardOutput.ReadToEndAsync();
        var stderrTask = process.StandardError.ReadToEndAsync();

        try
        {
            await process.WaitForExitAsync(timeoutCts.Token);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            _logger.LogWarning("Python script {Script} timed out after {TimeoutSeconds}s", SamplingScriptName, _options.TimeoutSeconds);
            TryKill(process);
            throw new TimeoutException($"Python script timed out after {_options.TimeoutSeconds} seconds.");
        }

        var stdout = await stdoutTask;
        var stderr = await stderrTask;

        _logger.LogInformation("Python script {Script} finished with exit code {ExitCode}",
            SamplingScriptName,
            process.ExitCode);

        return new ScriptRunResult(stdout, stderr, process.ExitCode);
    }

    private static void TryKill(Process process)
    {
        try
        {
            if (!process.HasExited)
            {
                process.Kill(entireProcessTree: true);
            }
        }
        catch (InvalidOperationException)
        {
        }
    }
}
