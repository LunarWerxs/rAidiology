# What a reading costs

The question this answers: *if rAidiology ever offered a paid reading to somebody with no computer
to run it on, what would each one cost to serve, so a price is arithmetic rather than a hope?*

Measured on 2026-10-01 on the development PC (`tools/bench_reading.py`, recorded in the run ledger
under `cost`). Host prices are list prices read the same day from each company's own pricing page;
every one is linked.

## The short answer

**An X-ray reading is about two and a half seconds of ordinary processor time, and a graphics card
is a waste of money on it.** On the route rAidiology actually ships it runs on the visitor's own
machine and costs us nothing at all. On a rented machine the compute for one reading is a
*hundredth of a cent*, which means the compute is never the thing that decides anything: the
monthly floor is.

## What was measured

One reading of an eight-image wrist series, real held-out radiographs, warm process, five-model
ensemble, decode included:

| | processor | graphics card |
|---|---|---|
| the whole reading | **2.45 s** | **0.91 s** |
| per image | 0.236 s | 0.045 s |
| peak memory | 1,954 MB with the CUDA build, **580 MB** with the processor-only build | 1,954 MB |
| starting cold (import plus weights) | 2.3 s | 2.3 s |

Three things in there are worth saying out loud:

- **More processor threads bought nothing.** One thread and eight threads came out the same, 0.238 s
  against 0.236 s an image. Do not pay for cores.
- **Decoding the pictures is a real part of the bill**, not a rounding error: turning one of a
  hospital's thousand-pixel PNGs into an array took 0.068 s cold, the same order as the entire
  five-model ensemble. Those figures include it, because the images are now decoded on four threads
  at once, measured at 1.91x (0.664 s to 0.348 s over twelve images).
- **Deploy the processor-only build.** The CUDA build of torch costs 1.4 GB of extra memory for
  nothing on a machine with no card, and memory is what picks your instance size.

## What that costs to rent

Per reading, on a host that bills by the second ([Modal](https://modal.com/pricing): $0.0000131 per
core-second, $0.00000222 per GiB-second, GPUs from $0.000222 a second for an L4):

| | compute per reading | readings per dollar |
|---|---|---|
| two cores, 2 GiB, processor only | **$0.000075** | about 13,000 |
| the same plus a cold start | $0.00018 | about 5,500 |
| an L4 graphics card | $0.00023 | about 4,300 |

So the graphics card costs about three times as much to go 2.7 times faster, on a job that already
finishes in two and a half seconds. There is no version of this where renting one for X-ray makes
sense.

On a host that bills by the month instead, the floor is the whole bill until the volume is serious.
The process needs about 600 MB, so the smallest instance is not big enough:

| host | smallest instance that fits | price |
|---|---|---|
| [Render](https://render.com/pricing) | 1 CPU / 2 GB (the 512 MB one is too small) | $25/month, plus the workspace plan: Hobby $0, Pro $25 |
| [Fly.io](https://fly.io/pricing/) | shared-cpu-4x with 1 GB | $7.78/month, RAM beyond that $5/GB/month |
| Modal | nothing to keep running: it scales to zero | $0, and $30/month of credit on the free plan |

Egress is not a factor for X-ray: the answer is a kilobyte, and the scan travels *inbound*. Render
charges $0.15/GB over its allowance, Fly $0.02/GB, Modal $0.04/GiB. A 300 MB CT would change that
conversation; an X-ray does not.

**Not read:** [Railway](https://railway.com/pricing)'s pricing page is behind a bot check and
RunPod's per-GPU table is drawn by script after the page loads, so neither was read and neither is
quoted here. Nothing in the conclusion depends on them.

## So what actually decides it

One core on a $25/month box serves roughly 1,400 readings an hour, which is about a million a month.
The compute for all of them is under a hundred dollars. What that means in practice:

| price per reading | readings a month to cover a $25 floor |
|---|---|
| $0.50 | 50 |
| $1.00 | 25 |
| $2.00 | 13 |

The cost of serving a reading is not the risk. **Nothing here says whether the paid route is a good
idea**; it says the compute bill will not be what makes or breaks it, and that if it is ever built,
it is built on processors, not on rented graphics cards.

## What is not measured, and would change the answer

- **CT and spine MRI.** The CT segmenter does not exist yet and spine MRI segmentation genuinely
  wants a graphics card. Those are the only parts of rAidiology where renting one could be the right
  call, and neither has been timed, because neither is finished. When the CT pipeline runs, this
  page gets its numbers.
- **A written AI reading.** The decision of 2026-10-01 is that the writing comes from the visitor's
  own AI on their own subscription, so there is no per-reading model bill. Bringing it in-house
  would add the only cost on this page that scales with every single reading.
- **Everything that is not compute.** Storage, a domain, somebody answering email when a person is
  frightened by what they read. Those are not on this page and they are not small.

## A reading is not identical on every machine

Measured the same day over twenty held-out radiographs (`tools/device_agreement.py`, recorded in the
ledger):

| | largest difference |
|---|---|
| the same image twice on the processor | **0** |
| the same image twice on the graphics card | **0** |
| processor against graphics card | **0.0090** |

The first two are zero because `detector.steady_inference()` pins cuDNN's algorithm choice; before
it, two runs of the same image on the same card disagreed. That matters more than it sounds:
somebody asking twice about their own wrist now gets the same answer twice, and it cost nothing
(0.071 s an image against 0.082).

The third one cannot be made zero. It is not a precision setting - the shipped path already runs in
full float32 with TF32 off - it is that different convolution kernels add things up in a different
order. What it means in practice, counted on the held-out test:

| | within 0.009 of the line |
|---|---|
| wrist (threshold 0.563) | 6 of 4,170 images, **0.14%** |
| mixed body parts (threshold 0.147) | 15 of 815 images, **1.84%** |

So about one wrist radiograph in 700 is close enough to the line that the kind of machine could
change its yes or no. Small, and worth writing down rather than discovering later: **the shipped
route reads on the processor**, which is what a laptop, a desktop and a rented container all do. If
a graphics-card server is ever added, its threshold has to be recalibrated on that hardware rather
than inherited, or it is a different detector wearing the same scorecard.
