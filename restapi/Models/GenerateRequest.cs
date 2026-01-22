using System.ComponentModel.DataAnnotations;

namespace WordStylist.Api.Models;

public sealed class GenerateRequest
{
    [Required]
    [MinLength(1)]
    public string Prompt { get; init; } = string.Empty;

    [Range(1, 500)]
    public int Steps { get; init; } = 50;

    [Range(0, int.MaxValue)]
    public int? Seed { get; init; }
}
