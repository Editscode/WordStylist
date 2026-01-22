using Microsoft.AspNetCore.Mvc;
using WordStylist.Api.Models;
using WordStylist.Api.Services;

namespace WordStylist.Api.Controllers;

[ApiController]
[Route("api/[controller]")]
public sealed class GenerateController : ControllerBase
{
    private readonly IPythonScriptRunner _runner;
    private readonly ILogger<GenerateController> _logger;

    public GenerateController(IPythonScriptRunner runner, ILogger<GenerateController> logger)
    {
        _runner = runner;
        _logger = logger;
    }

    [HttpPost]
    public async Task<IActionResult> GenerateAsync([FromBody] GenerateRequest request, CancellationToken cancellationToken)
    {
        var result = await _runner.RunSamplingAsync(request, cancellationToken);

        if (result.ExitCode != 0)
        {
            _logger.LogWarning("Python script returned non-zero exit code {ExitCode}", result.ExitCode);
            return StatusCode(StatusCodes.Status500InternalServerError, new
            {
                result.ExitCode,
                result.Stdout,
                result.Stderr
            });
        }

        return Ok(new
        {
            result.ExitCode,
            result.Stdout,
            result.Stderr
        });
    }
}
