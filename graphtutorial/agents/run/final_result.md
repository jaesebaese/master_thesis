# Benchmark Compliance

| Setting | Configured | Recommended | Status | Policy Name |
|---|---:|---|---|---|
| Minimum Password Age | 1 | 1 | COMPLIANT | CIS - Device Lock [L1] - Windows 11 - v4.0.0.0 |
| Do not allow passwords to be saved | Enabled | 1 | COMPLIANT | CIS - Windows Components - Restrictions [L1] - Windows 11 - v4.0.0.0 |
| Password Length | 10 | 14 | NON-COMPLIANT | CIS - Windows LAPS [L1] - Windows 11 - v4.0.0.0 |
| Password Complexity | Large letters + small letters + numbers + special characters |  | NOT IN BENCHMARK | CIS - Windows LAPS [L1] - Windows 11 - v4.0.0.0 |
| Device Password History | 24 | 24 | COMPLIANT | CIS - Device Lock [L1] - Windows 11 - v4.0.0.0 |
| Password Age Days | 30 | 365 | COMPLIANT | CIS - Windows LAPS [L1] - Windows 11 - v4.0.0.0 |
| Device Password Expiration | 365 | 365 | COMPLIANT | CIS - Device Lock [L1] - Windows 11 - v4.0.0.0 |
| Account Logon Logoff Audit Account Lockout | Failure | 2 | NOT IN BENCHMARK | CIS - Auditing [L1] - Windows 11 - v4.0.0.0 |
| Max Inactivity Time Device Lock | 15 | 15 | COMPLIANT | CIS - Device Lock [L1] - Windows 11 - v4.0.0.0 |
| Idle session limit: (Device) | 15 minutes | 15 minutes | COMPLIANT | CIS - Windows Components - Restrictions [L2] - Windows 11 - v4.0.0.0 |
| Set time limit for disconnected sessions | Enabled |  | NOT IN BENCHMARK | CIS - Windows Components - Restrictions [L2] - Windows 11 - v4.0.0.0 |
| Interactive Logon Machine Inactivity Limit | 900 | 900 | COMPLIANT | CIS - Local Policies Security Options [L1] - Windows 11 - v4.0.0.0 |
| Set time limit for active but idle Remote Desktop Services sessions | Enabled |  | NOT IN BENCHMARK | CIS - Windows Components - Restrictions [L2] - Windows 11 - v4.0.0.0 |
| End a disconnected session (Device) | 1 minute |  | NOT IN BENCHMARK | CIS - Windows Components - Restrictions [L2] - Windows 11 - v4.0.0.0 |

## Remediation

- **Password Length** — CIS remediation: increase Windows LAPS password length to **14**.

## Interdependencies

- No structural conflicts or unmet prerequisites were reported by the interdependency analysis.

## Web Search Results

Rows below used web search because they were not covered by the CIS benchmark:

- **Password Complexity** — Microsoft guidance was returned for Windows LAPS complexity; lower values are documented as backwards-compatibility settings only.
- **Account Logon Logoff Audit Account Lockout** — no Microsoft recommendation returned.
- **Set time limit for disconnected sessions** — no Microsoft recommendation returned.
- **Set time limit for active but idle Remote Desktop Services sessions** — no Microsoft recommendation returned.
- **End a disconnected session (Device)** — no Microsoft recommendation returned.

## Security Requirements Compliance

| Requirement id | Requirement Description | Expected Value | Addressed by Settings | Set Value | Compliance Status |
|---|---|---:|---|---|---|
| REQ-001 | Passwords must be created to ensure they are difficult to guess or compromise. |  | Minimum Password Age; device_vendor_msft_policy_config_devicelock_mindevicepasswordcomplexcharacters; device_vendor_msft_policy_config_devicelock_alphanumericdevicepasswordrequired; device_vendor_msft_policy_config_credentialsui_disablepasswordreveal | 1; 2; 0; 1 | SATISFIED |
| REQ-002 | All passwords must be unique to each system and must not be reused across different services. |  | device_vendor_msft_policy_config_userrights_denylogonasservice | ['*S-1-5-32-546'] | NOT COVERED |
| REQ-003 | Passwords must remain confidential and must not be shared with any other individual, including IT personnel. |  | Do not allow passwords to be saved | Enabled | NOT COVERED |
| REQ-004 | Passwords must not be stored in plain text (digitally or physically) unless protected by approved secure storage mechanisms such as password managers. | false | Do not allow passwords to be saved | Enabled | SATISFIED |
| REQ-005 | Passwords must have a minimum length of twelve characters. | 12 | Password Length | 10 | VIOLATED |
| REQ-006 | Passwords must include a combination of uppercase letters, lowercase letters, numbers, and special characters. |  | Password Complexity | Large letters + small letters + numbers + special characters | SATISFIED |
| REQ-007 | Where systems allow, passphrases should be at least sixteen characters and composed of multiple unrelated words combined with numbers or special characters. | 16 |  |  | NOT COVERED |
| REQ-008 | Users must not reuse previous passwords; the organization will enforce controls to prevent reuse of at least the last ten passwords. | 10 | Device Password History; Notify Password Reuse | 24; Enabled | SATISFIED |
| REQ-009 | Standard user accounts must update their passwords at least every ninety days. | 90 | Device Password Expiration | 365 | VIOLATED |
| REQ-010 | Privileged accounts must be updated every sixty days. | 60 | Password Age Days | 30 | SATISFIED |
| REQ-011 | Passwords must be changed immediately if there is any suspicion that they have been compromised. |  | Minimum Password Age | 1 | VIOLATED |
| REQ-012 | Multi-factor authentication is required for access to sensitive systems, remote access services, and all privileged accounts. | true |  |  | NOT COVERED |
| REQ-013 | Accounts will be automatically locked after five failed login attempts. | 5 | Account Logon Logoff Audit Account Lockout; Max device password failed attempts | Failure; 5 | NOT COVERED |
| REQ-014 | After lockout, the account will be locked for a minimum of fifteen minutes. | 15 |  |  | NOT COVERED |
| REQ-015 | User sessions must be automatically terminated after fifteen minutes of inactivity. | 15 | Idle session limit: (Device); Interactive Logon Machine Inactivity Limit | 15 minutes; 900 | SATISFIED |
| REQ-016 | Passwords must not be stored in plain text; they must be protected using strong cryptographic hashing algorithms such as bcrypt or Argon2. |  |  |  | NOT COVERED |
| REQ-017 | Passwords must only be transmitted over secure channels that provide encryption. | true |  |  | NOT COVERED |
| REQ-018 | Privileged accounts must use stronger passwords with a minimum length of sixteen characters. | 16 | Password Length | 10 | VIOLATED |
| REQ-019 | Shared use of privileged accounts is not permitted. | false |  |  | NOT COVERED |

## Security Posture Summary

The tenant shows partial alignment with the provided security policy. CIS-covered controls are mostly in place, but several policy requirements are not covered by the current configuration set, and the most significant CIS-aligned gaps are password length for privileged/local admin passwords, standard password expiration, and immediate password-change expectations after suspected compromise. Microsoft guidance was only available for a small subset of the settings not covered by CIS. Overall posture is mixed: foundational device/session controls are present, but password lifecycle and identity assurance requirements remain incomplete.
