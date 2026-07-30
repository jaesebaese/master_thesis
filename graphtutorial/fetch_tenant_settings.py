# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# <ProgramSnippet>
import asyncio
import configparser
import os
from graph import Graph

async def main():
    print('Python Graph Tutorial\n')

    # Load settings
    config = configparser.ConfigParser()
    config_dir = os.path.dirname(__file__)
    config.read([os.path.join(config_dir, 'config.cfg'), os.path.join(config_dir, 'config.dev.cfg')])
    azure_settings = config['azure']

    graph: Graph = Graph(azure_settings)

    await greet_user(graph)

    await make_graph_call(graph)

# <GreetUserSnippet>
async def greet_user(graph: Graph):
    print('Hello :)')
# </GreetUserSnippet> 


# <MakeGraphCallSnippet>
async def make_graph_call(graph: Graph):
    policies = await graph.make_graph_call()
    if policies:
        # Output each policy's details
        for policy in policies:
            print('Policy:', policy.get('displayName') or policy.get('name') or 'Unknown')
            print('  ID:', policy.get('id'))
            print('  Description:', policy.get('description') or 'None')
            # Add more fields if needed
        print(f'\nTotal policies retrieved: {len(policies)}\n')
    else:
        print('No policies found.\n')
# </MakeGraphCallSnippet>

# Run main
asyncio.run(main())
