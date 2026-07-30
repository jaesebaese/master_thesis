# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import requests
import json
from configparser import SectionProxy
from azure.identity import DeviceCodeCredential
from msgraph import GraphServiceClient
from msgraph.generated.users.item.user_item_request_builder import UserItemRequestBuilder
import requests

class Graph:
    settings: SectionProxy
    device_code_credential: DeviceCodeCredential
    user_client: GraphServiceClient

    def __init__(self, config: SectionProxy):
        self.settings = config
        client_id = self.settings['clientId']
        tenant_id = self.settings['tenantId']
        graph_scopes = self.settings['graphUserScopes'].split(' ')

        self.device_code_credential = DeviceCodeCredential(client_id, tenant_id = tenant_id)
        self.user_client = GraphServiceClient(self.device_code_credential, graph_scopes)
# </UserAuthConfigSnippet>

    # <GetUserTokenSnippet>
    async def get_user_token(self):
        graph_scopes = self.settings['graphUserScopes']
        access_token = self.device_code_credential.get_token(graph_scopes)
        return access_token.token
    # </GetUserTokenSnippet>

    # <GetUserSnippet>
    async def get_user(self):
        # Only request specific properties using $select
        query_params = UserItemRequestBuilder.UserItemRequestBuilderGetQueryParameters(
            select=['displayName', 'mail', 'userPrincipalName']
        )

        request_config = UserItemRequestBuilder.UserItemRequestBuilderGetRequestConfiguration(
            query_parameters=query_params
        )

        user = await self.user_client.me.get(request_configuration=request_config)
        return user
    # </GetUserSnippet>

    def get_policy_with_settings(self, token, policy_id):
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        url = f"https://graph.microsoft.com/beta/deviceManagement/configurationPolicies/{policy_id}/settings"
        
        settings = []
        while url:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            settings.extend(data.get("value", []))
            url = data.get("@odata.nextLink")
    
        return settings
    
    def get_policy_with_assignments(self, token, policy_id):
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        url = f"https://graph.microsoft.com/beta/deviceManagement/configurationPolicies/{policy_id}/assignments"
        
        assignments = []
        while url:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            assignments.extend(data.get("value", []))
            url = data.get("@odata.nextLink")
    
        return assignments

    # <MakeGraphCallSnippet>
    async def make_graph_call(self):
        token = await self.get_user_token()
        url = "https://graph.microsoft.com/beta/deviceManagement/configurationPolicies?$expand=settings,assignments"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

        all_policies = []
        while url:
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            all_policies.extend(data.get("value", []))
            url = data.get("@odata.nextLink")
        

        print(all_policies)
        
        """all_policies_with_settings_assigns = []

        for policy in all_policies:
            settings = self.get_policy_with_settings(token, policy.get("id"))
            assignments = self.get_policy_with_assignments(token, policy.get("id"))
            policy["settings"] = settings
            policy["assignments"] = assignments
            all_policies_with_settings_assigns.append(policy)

 """
        
    async def fetch_all_configuration_settings(self):
        token = await self.get_user_token()
        url = "https://graph.microsoft.com/beta/deviceManagement/configurationSettings"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

        all_policies = []
        while url:
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            all_policies.extend(data.get("value", []))
            url = data.get("@odata.nextLink")
        

        print(all_policies[:20])
        # Write all_policies to a new file
        with open('microsoft_settings.json', 'w') as f:
           json.dump(all_policies, f, indent=4)

        return all_policies
    # </MakeGraphCallSnippet>

