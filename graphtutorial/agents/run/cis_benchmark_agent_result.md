### Tenant vs CIS Benchmark (focus: CIS-compliance only)

| Requirement ID | CIS ID | Setting | Configured Value | CIS Recommended | Status |
|---|---|---|---|---|---|
| REQ-001 | 26.3 | device_vendor_msft_policy_config_devicelock_mindevicepasswordcomplexcharacters | 2 | 2 | COMPLIANT |
| REQ-005 | 26.8 | device_vendor_msft_policy_config_devicelock_mindevicepasswordlength | 14 | 14 | COMPLIANT |
| REQ-001 | 26.9 | device_vendor_msft_policy_config_devicelock_minimumpasswordage | 1 | 1 | COMPLIANT |
| REQ-001 | 26.2 | device_vendor_msft_policy_config_devicelock_alphanumericdevicepasswordrequired | 0 | 0 | COMPLIANT |
| REQ-001 | 4.11.8.1 | device_vendor_msft_policy_config_credentialsui_disablepasswordreveal | 1 | 1 | COMPLIANT |
| REQ-002 | 89.16 | device_vendor_msft_policy_config_userrights_denylogonasservice | `['*S-1-5-32-546']` | `['*S-1-5-32-546']` | COMPLIANT |
| REQ-003 | 4.11.36.3.2 | device_vendor_msft_policy_config_remotedesktopservices_donotallowpasswordsaving | 1 | 1 | COMPLIANT |
| REQ-004 | 76.1.3 | device_vendor_msft_policy_config_webthreatdefense_notifyunsafeapp | 1 | 1 | COMPLIANT |
| REQ-008 | 26.5 | device_vendor_msft_policy_config_devicelock_devicepasswordhistory | 24 | 24 | COMPLIANT |
| REQ-008 | 26.4 | device_vendor_msft_policy_config_devicelock_devicepasswordexpiration | 365 | 365 | COMPLIANT |
| REQ-008 | 76.1.2 | device_vendor_msft_policy_config_webthreatdefense_notifypasswordreuse | 1 | 1 | COMPLIANT |
| REQ-009 | 26.9 | device_vendor_msft_policy_config_devicelock_minimumpasswordage | 1 | 1 | COMPLIANT |
| REQ-010 | 26.4 | device_vendor_msft_policy_config_devicelock_devicepasswordexpiration | 365 | 365 | COMPLIANT |
| REQ-013 | 6.2 | device_vendor_msft_policy_config_audit_accountlogonlogoff_auditaccountlockout | 2 | 2 | COMPLIANT |
| REQ-013 | 26.6 | device_vendor_msft_policy_config_devicelock_maxdevicepasswordfailedattempts | 5 | 5 | COMPLIANT |
| REQ-014 | 26.7 | device_vendor_msft_policy_config_devicelock_maxinactivitytimedevicelock | 15 | 15 | COMPLIANT |
| REQ-015 | 49.8 | device_vendor_msft_policy_config_localpoliciessecurityoptions_interactivelogon_machineinactivitylimit_v2 | 900 | 900 | COMPLIANT |

#### NON-COMPLIANT items
- None found (0).

#### NOT IN BENCHMARK items encountered (configured in tenant, but CIS has no reference for them)
- These settings were found via tenant search and do not map to a CIS benchmark reference value:
  - device_vendor_msft_laps_policies_passwordlength (Configured: 10)
  - device_vendor_msft_laps_policies_passwordcomplexity (Configured: `device_vendor_msft_laps_policies_passwordcomplexity_4`)
  - device_vendor_msft_laps_policies_passwordagedays_aad (Configured: 30)
  - device_vendor_msft_policy_config_admx_terminalserver_ts_sessions_idle_limit_2_ts_sessions_idlelimittext (Configured: `..._900000`)
  - device_vendor_msft_policy_config_admx_terminalserver_ts_sessions_disconnected_timeout_2 (Configured: `..._1`)
  - device_vendor_msft_policy_config_admx_terminalserver_ts_sessions_idle_limit_2 (Configured: `..._1`)
  - device_vendor_msft_policy_config_admx_terminalserver_ts_sessions_disconnected_timeout_2_ts_sessions_enddisconnected (Configured: `..._60000`)

### Notes on MFA / Session termination / other CIS-mapped requirements
- The benchmark result indicates some requirements had no CIS matches (e.g., MFA requirement, some password hashing/transmission requirements), and therefore no CIS setting comparison could be performed for those items based on the benchmark dataset returned.