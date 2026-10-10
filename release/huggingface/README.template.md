---
license: other
license_name: polyform-noncommercial-1.0.0
license_link: LICENSE
tags:
  - medical-imaging
  - not-a-medical-device
extra_gated_heading: "Accept the licence to download this model"
extra_gated_description: "Access is granted automatically once you accept. Hugging Face shares your username and email address with LunarWerx when you ask for access."
extra_gated_prompt: |-
  This model is released under the PolyForm Noncommercial License 1.0.0. The full text is in the LICENSE file of this repository. To download it, you agree to the following.

  1. Noncommercial use only. A person may use the model for personal study and for research with no expected commercial use. A charity, school or university, public research body, public safety or health body, environmental protection body or government body may use it. Any other use is commercial and needs a licence from LunarWerx.
  2. Not a medical device. The model is not cleared or approved by the FDA or any other regulator. You will not use it to diagnose or treat anybody, or to make decisions about a patient or about your own health. Talk to a doctor about any scan result.
  3. No re-identification. You will not use the model or its outputs to try to find out who a person is.
  4. Commercial use needs a licence. To use the model in paid work, in a product or in a service, ask LunarWerx for a commercial licence first. You can reach LunarWerx at contact@lunarwerx.com.
  5. Your details. The details you enter below go to LunarWerx.
extra_gated_fields:
  Name: text
  Affiliation: text
  Country: country
  Intended use:
    type: select
    options:
      - Research
      - Education
      - Personal
      - Other
  I have read the PolyForm Noncommercial License 1.0.0 in this repository: checkbox
  I will use this model for noncommercial purposes only: checkbox
  I understand this model is not a medical device and I will not use it for diagnosis or treatment: checkbox
  I will not use this model or its outputs to try to identify any person: checkbox
  I understand that commercial use needs a separate licence from LunarWerx: checkbox
extra_gated_button_content: "I agree and ask for access"
---

{{CARD}}
