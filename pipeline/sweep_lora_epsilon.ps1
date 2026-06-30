param(
    [int]$Rounds = 10,
    [int]$FastDevRun = 50,
    [int]$Clients = 2,
    [int]$BatchSize = 32
)

$env:PYTHONIOENCODING = "utf-8"
$base = "model.pretrained=true model.use_lora=true dataset.num_clients=$Clients training.num_rounds=$Rounds training.local_epochs=1 training.batch_size=$BatchSize training.fast_dev_run=$FastDevRun dataset.alpha=0.5"

$runs = @(
    @{ name = "lora_fedavg";     extra = "privacy=none" },
    @{ name = "lora_sigma030";   extra = "privacy=dp_sgd privacy.noise_multiplier=0.3" },
    @{ name = "lora_sigma110";   extra = "privacy=dp_sgd privacy.noise_multiplier=1.1" },
    @{ name = "lora_sigma300";   extra = "privacy=dp_sgd privacy.noise_multiplier=3.0" }
)

foreach ($run in $runs) {
    Write-Host "`n====== Starting: $($run.name) ======" -ForegroundColor Cyan
    $cmd = "& 'C:\Users\user\.local\bin\uv.exe' run python pipeline/train.py $base experiment.name=$($run.name) $($run.extra)"
    Write-Host $cmd
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $($run.name) (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "DONE: $($run.name)" -ForegroundColor Green
}

Write-Host "`n====== All LoRA sweep runs complete ======" -ForegroundColor Cyan
