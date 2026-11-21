# Magic Greetings — release check, 21 November 2026

Status: catalogue and portrait-template changes deployed; NOT cleared for commercial release.

## Completed
- Preserved the four approved compositions: Christmas Market, The Last Ornament, Enchanted Winter Forest (corrected waist-up couple), Christmas Morning.
- Added Velvet Christmas Portrait using the exact header artwork, with a locally enforced face mask that protects clothing, pose, lettering, lantern, frame and background.
- Replaced four cramped samples: City of Christmas Lights, By the Christmas Fireplace, Christmas Forest Lantern, New Year City Fireworks.
- Reviewed Snowy Winter Forest and Winter City Panorama; retained the visible surrounding scenery.
- Selection callback, checkmarked button labels, green selected-card outline and toast.
- Generation stages and disabled busy button; bounded face detection input; environment-focused prompt and smaller input placement for newly generated scenes.

## Validation
12 unit tests pass. Controlled-provider integration tests pass for generation, automatic recovery, PNG rendering/download, one-face refinement and HTTP 401/429 quota/503 messages.
Royal-template integration passes with real template detection, face-only mask, original-reference handling and PNG download.
Live website loads the updated gallery; portrait selection visibly becomes checked and the form appears.
All catalogue people are fictional. Header artwork remains in place.

## Unresolved blockers
No successful real paid API image request was observed. The older reported ten-minute stall has no confirmed server-log diagnosis.
Secure browser credential collection was automatically rejected because the trial-generation password is not a conventional account login. No credential workaround was attempted.
Need a redacted Streamlit server error/request log or an authorized supported diagnostic route for actual API verification.
Commercial payment, post-payment access, correction allowance enforcement, failure handling and end-to-end paid delivery must be verified before accepting customers.

## Target
Aim to complete the agreed scope within the user's requested 2–3 days (23–24 November). This is a target, not an unverified promise of readiness.
Code commit: 2636e4ba3ba616ba6429d8bc890cf1c6522f4c92.
