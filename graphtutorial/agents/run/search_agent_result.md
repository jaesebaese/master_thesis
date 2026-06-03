## Settings not covered by CIS benchmark (per search results)

- **Setting ID**: `device_vendor_msft_laps_policies_passwordlength`  
  - **Description**: Windows LAPS generates managed local admin passwords with a configurable length. Microsoft Learn describes the default and configurable range.  
  - **Microsoft Recommendation**: Use Windows LAPS to automatically manage local admin passwords and configure the password length to your requirements. Microsoft notes the default and the allowed range.  
  - **Recommended Value**: Default is **14 characters**; can be configured **8–64 characters**.  
  - **Source**: https://learn.microsoft.com/en-us/windows-server/identity/laps/laps-concepts-passwords-passphrases

- **Setting ID**: `device_vendor_msft_laps_policies_passwordcomplexity`  
  - **Description**: Microsoft documents that Windows LAPS supports lower password complexity values only for compatibility with older LAPS versions.  
  - **Microsoft Recommendation**: Avoid relying on lower complexity settings except for backwards compatibility with older LAPS; use appropriate complexity settings supported by your environment.  
  - **Recommended Value**: Lower password complexity settings **(1, 2, and 3)** are supported only for **backwards compatibility**.  
  - **Source**: https://learn.microsoft.com/en-us/windows/client-management/mdm/laps-csp

- **Setting ID**: `device_vendor_msft_laps_policies_passwordagedays_aad`  
  - **Description**: Microsoft Intune’s Windows LAPS support focuses on backing up managed local admin accounts/passwords to **Microsoft Entra ID** (cloud) and notes manual rotation capability.  
  - **Microsoft Recommendation**: Use Intune’s Windows LAPS endpoint security policy to back up local admin account passwords to **Entra ID**, and you can also manually rotate passwords outside scheduled rotation.  
  - **Recommended Value**: *No specific numeric age/days value found in the returned Microsoft search results.*  
  - **Source**: https://learn.microsoft.com/en-us/intune/device-security/laps/overview

- **Setting ID**: `device_vendor_msft_policy_config_admx_terminalserver_ts_sessions_idle_limit_2_ts_sessions_idlelimittext`  
  - **Note**: No Microsoft documentation found — human review required.

- **Setting ID**: `device_vendor_msft_policy_config_admx_terminalserver_ts_sessions_disconnected_timeout_2`  
  - **Note**: No Microsoft documentation found — human review required.

- **Setting ID**: `device_vendor_msft_policy_config_admx_terminalserver_ts_sessions_idle_limit_2`  
  - **Note**: No Microsoft documentation found — human review required.

- **Setting ID**: `device_vendor_msft_policy_config_admx_terminalserver_ts_sessions_disconnected_timeout_2_ts_sessions_enddisconnected`  
  - **Note**: No Microsoft documentation found — human review required.