using WordStylist.Api.Options;
using WordStylist.Api.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

builder.Services.Configure<ScriptRunnerOptions>(
    builder.Configuration.GetSection(ScriptRunnerOptions.SectionName));

builder.Services.AddScoped<IPythonScriptRunner, PythonScriptRunner>();

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

app.MapControllers();

app.Run();
