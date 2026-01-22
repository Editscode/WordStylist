namespace WordStylist.Api.Models;

public sealed class ScriptRunResult
{
    public ScriptRunResult(string stdout, string stderr, int exitCode)
    {
        Stdout = stdout;
        Stderr = stderr;
        ExitCode = exitCode;
    }

    public string Stdout { get; }

    public string Stderr { get; }

    public int ExitCode { get; }
}
