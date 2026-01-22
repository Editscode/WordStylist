using WordStylist.Api.Models;

namespace WordStylist.Api.Services;

public interface IPythonScriptRunner
{
    Task<ScriptRunResult> RunSamplingAsync(GenerateRequest request, CancellationToken cancellationToken);
}
