import time, torch

def benchmark_training(model, train_loader, opt, sched, loss_fn, cfg, device,
                       warmup=10, timed=50):
    """
    Run a short benchmark to estimate throughput and training time.
    """
    model.train()
    i, start = 0, None

    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        if i == warmup:
            torch.cuda.synchronize()
            start = time.perf_counter()

        # one full training step
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            logits = model(xb)
            loss = loss_fn(cfg, logits, yb)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()

        i += 1
        if i >= warmup + timed:
            break

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    iters_per_sec = timed / elapsed

    steps_per_epoch = len(train_loader)
    secs_per_epoch  = steps_per_epoch / iters_per_sec
    total_secs      = cfg.epochs * secs_per_epoch

    print(f"[benchmark] {iters_per_sec:.2f} it/s (training step)")
    print(f"[estimate] {secs_per_epoch/60:.1f} min/epoch, "
          f"{total_secs/3600:.1f} h total for {cfg.epochs} epochs")