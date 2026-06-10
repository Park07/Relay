# Chaos tests (need the live stack)

Fault-injection scenarios that validate the reliability claims (DESIGN.md §5.2,
ADR-6). They require the compose stack and deliberately perturb it:

- Worker crash mid-batch → lease stream drops → `XAUTOCLAIM` reassigns the
  pending entries after the lease timeout; the client still gets a result.
- Redis blip → gateway `/readyz` 503s, scheduler reconnects, no lost acks.
- Backpressure: drive offered load past the high-water mark and assert the
  gateway sheds with 429 instead of growing an unbounded queue.
- Autoscale: ramp `relay_queue_depth` and observe the HPA add worker pods, then
  scale back down when the queue drains.
