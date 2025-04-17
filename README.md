AGI-in-a-box

Exactly what it sounds like: an LLM based collection of agent workflows that performs some of the tasks an AGI 
might need.  The title is pretentious, but the work is unglamorous.

Those tasks include things like:
  - self-configuration (change Ansible variable values on the fly and reconfigured the install)
  - determine if exterior cloud-based agents need to run to accomplish a task, and execute those agents
  - commit existing request history to a git repository automatically to provide a complete history of use
  - backup existing repository history past a specific threshold to low cost cloud backup target
  - backup existing model history, if desired, to a low cost cloud backup target for reuse
  - determine if new models should be downloaded and used in place of existing models, and obtain them
  - write LLM chat output to specific files for reuse, as needed, to reconfigure the main application
  - maintain a dynamic list of data sources and files to ingest via RAG for given tasks
  - maintain a prompt library for easy retrieval and reuse via Ansible variables and version control
  - methods for transferring the application to new hardware for upgrade or expansion

Why build this??
  - why not?
  - hardware configuration and orchestration keeps AGI practical and yours
  - as you use AI, the knowledge grows but so does the data and history
  - This can act as a model for other large-scale open source AI apps

Prerequisites:
  - install poetry to build an isolated environment for dependencies
  - run "poetry init" to install those dependencies
  - to run any of the agent workflows, type "poetry run python {name of script}"

Right now, these workflows in here are just tests so I can understand how agentic frameworks work.
