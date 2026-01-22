namespace WordStylist.Api.Options;

public sealed class ScriptRunnerOptions
{
    public const string SectionName = "ScriptRunner";

    public string PythonPath { get; init; } = "python";

    public string ScriptsRoot { get; init; } = "/workspace/WordStylist";

    public int TimeoutSeconds { get; init; } = 300;
}
