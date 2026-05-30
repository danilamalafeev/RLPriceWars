$ErrorActionPreference = 'Stop'
Set-Location 'C:\Users\я\Documents\RLpricewars'
$root = 'C:\Users\я\Documents\RLpricewars\results\long_matrix_100k_plus\block1_static_victim_100k'
$runnerLog = 'C:\Users\я\Documents\RLpricewars\results\long_matrix_100k_plus\block1_static_victim_100k\static_runner.log'
$tasks = @()
foreach ($mode in @('dqn','tabular_cfr')) {
  foreach ($seed in 0..9) {
    $outDir = Join-Path $root (Join-Path $mode "seed_$seed")
    $tasks += [pscustomobject]@{ mode=$mode; seed=$seed; outDir=$outDir }
  }
}
$running = @()
$maxParallel = 4
function Start-StaticTask($task) {
  $logDir = Join-Path $task.outDir 'logs'
  New-Item -ItemType Directory -Force -Path $logDir | Out-Null
  $stdout = Join-Path $logDir 'stdout.log'
  $stderr = Join-Path $logDir 'stderr.log'
  $env:OMP_NUM_THREADS = if ($task.mode -eq 'dqn') { '2' } else { '1' }
  $env:MKL_NUM_THREADS = if ($task.mode -eq 'dqn') { '2' } else { '1' }
  $args = @('-m','experiments.dqn_oracle_vs_qvictim','--oracle-kind',$task.mode,'--victim-kind','static_cooperative','--seed',[string]$task.seed,'--total-steps','100000','--eval-every','5000','--eval-steps','2000','--out-dir',$task.outDir)
  Start-Process -FilePath 'python' -ArgumentList $args -WorkingDirectory 'C:\Users\я\Documents\RLpricewars' -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
}
foreach ($task in $tasks) {
  $summary = Join-Path $task.outDir 'summary.json'
  if (Test-Path $summary) {
    "SKIP static $($task.mode) seed=$($task.seed)" | Add-Content -Path $runnerLog
    continue
  }
  while ($running.Count -ge $maxParallel) {
    Wait-Process -Id ($running | Select-Object -First 1).Id
    $running = @($running | Where-Object { -not $_.HasExited })
  }
  "RUN static $($task.mode) seed=$($task.seed)" | Add-Content -Path $runnerLog
  $running += Start-StaticTask $task
}
foreach ($p in $running) { Wait-Process -Id $p.Id }
"static victim control completed" | Add-Content -Path $runnerLog
