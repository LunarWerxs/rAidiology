# Contributing to rAidiology

Thank you for wanting to help. This page says what we take, what we do not take, and how a change gets in.

rAidiology is not a medical device. It is for education and research, never for diagnosis or treatment decisions ([LICENSING.md](LICENSING.md#not-a-medical-device)).

## The short version

- Send code, training recipes, configs, evaluation scripts and fixes to the docs.
- Never send model weights. Never send scans or any data. A pull request that carries a weight file or a scan is closed.
- Sign the contributor licence agreement once. A bot asks on your first pull request.
- Keep patient data out of everything: code, tests, issues and screenshots.

## What a pull request may contain

Welcome:

- code (readers, pipelines, the viewer, the MCP server)
- training recipes and configs
- evaluation scripts and fixes to how a score is worked out
- tests
- fixes to documentation

Not accepted, and closed without review:

- weight files (`.pt`, `.pth`, `.onnx`, `.safetensors`, `.bin`, or any other format)
- scans or images of people: DICOM, NIfTI, PNG, JPEG and similar, in any form, even zipped
- datasets, or files made from a dataset

If your change needs a trained model, send the recipe that trains it. Do not send the model.

## Why weights stay with the maintainer

LunarWerx (the maintainer) retrains every model itself and publishes the official weights itself. We do this so that every published model:

- comes from a recipe anybody can read,
- was trained only on data we are allowed to use,
- was scored the same way as the model before it.

So your change reaches users after the maintainer retrains with it. It does not reach them as a file you attach.

## How a change is judged

The rules are the same for everyone, including the maintainer. They are written down in [modalities/README.md](modalities/README.md).

- Changes are built and compared on the development folds only, by cross-validation, with patient-level 95% confidence intervals.
- A test set is locked away and opened once. A model ships only if, on that test, it beats the best baseline and the 95% confidence interval of the difference is above zero.
- Scores are recomputed by a separate script, so the numbers do not rest on one person's word.

If you propose a change to a model, say how you think it will move the development score. If you have run it, show the numbers and how you got them. Do not fit anything to a test set.

## Data

- Never include patient data. That means no scans, no reports, no names, no dates of birth, no IDs, and no screenshots that show any of them.
- Use public datasets only, and only when their licence allows commercial use. This is the project's rule, because LunarWerx sells commercial licences ([LICENSING.md](LICENSING.md)).
- Do not add data files to a pull request. Link to the dataset's page.
- If your change uses a dataset that is not in [LICENSING.md](LICENSING.md), say in the pull request: its name, where it is published, its licence (with a link to where the licence is stated), and the credit it asks for. The maintainer checks it and adds it to LICENSING.md before anybody trains on it.

## The contributor licence agreement

Every contributor signs the [contributor licence agreement](CLA.md) once.

- You keep the copyright in your work.
- You give LunarWerx the right to use it, change it and license it onward under any terms, including paid commercial licences. LunarWerx sells those licences, and needs this right to do so.
- You promise the work is yours to give, and that it contains no patient data and no code under a licence that does not fit.

When you open your first pull request, a bot (CLA Assistant Lite) comments on it with a link to the agreement. If you agree, reply to the pull request with the sentence the comment gives you. After that, you are covered for later pull requests. The pull request cannot be merged until you have signed.

Read the agreement before you sign it.

## How to send a pull request

1. Fork the repository and make a branch.
2. Keep the change small. One change per pull request is easier to read and easier to merge.
3. Say what it does and why. Say how you checked it. If it changes a score, show the before and after.
4. List anything you copied from somewhere else, with its licence.
5. Be kind in comments. Disagree about the work, not the person.

The maintainer may ask for changes, may decline a pull request, or may take your approach and build it in a different form. Declining a pull request is not a judgement of you.

## Your contribution and the licence

The project is under the [PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use needs a separate licence from LunarWerx. When you contribute, your work becomes part of the project and is offered under those terms, together with the grant you make in the [CLA](CLA.md).

## Reporting a problem

Open an issue on the repository's Issues page: https://github.com/LunarWerxs/rAidiology/issues

Please include:

- what you ran, and the command or page you used,
- what you expected,
- what happened instead, with the exact error text,
- the version or commit, and your operating system.

If the problem is a wrong or missed finding, give a public example: the dataset's name and the case's ID. Do not attach your own scan, or anyone else's, to an issue. This holds for screenshots too, because a name or date can sit in a corner of the picture.

Please do not ask for medical advice in an issue. rAidiology cannot give it. Talk to a doctor about your results.

If you find a security problem or a privacy leak (for example, a file that seems to hold patient data), do not post the details in public. Use the Security tab of the repository to report it privately if it is offered. If it is not, open an issue that says only that you have a security report, with no details, and a maintainer will contact you. You can also reach LunarWerx at contact@lunarwerx.com.
