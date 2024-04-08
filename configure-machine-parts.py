from crewai import Crew, Process
from crewai import Task
from crewai_tools import SerperDevTool
from crewai import Agent
import os

os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
# Adjust based on available model"
os.environ["OPENAI_MODEL_NAME"] = "mixtral"
# os.environ["OPENAI_API_KEY"] = <insert OpenAI API key here >
# os.environ["SERPER_API_KEY"] = <insert Serper API key here >

search_tool = SerperDevTool()


# Creating a systems architect agent with memory and verbose mode
researcher = Agent(
    role="AI/ML Chief Architect",
    goal=(
        "Assemble a parts list for a {purpose} system for {budget} USD"
        "capable of running LLM models greater than 13B parameters"
    ),
    verbose=True,
    memory=True,
    backstory=(
        "Working with an AI startup to develop LLM augmented software services"
        "and requiring 100% self hosted hardware for customer compliance"
        "reasons. This machine needs to be able to run in a commercial office"
        "with standard 120V 10A power, and be quiet enough to not disturb other"
        "office tenants unless it runs at night."
    ),
    tools=[search_tool],
    allow_delegation=True,
)

# Creating a writer agent with custom tools and delegation capability
writer = Agent(
    role="Purchaser",
    goal="Find parts for the requested machine under {budget}",
    verbose=True,
    memory=True,
    backstory=(
        "Search for compatible parts on auction sites, consumer PC parts"
        "stores, and vendor websites.  These parts need to all be compatible"
        "with each other (use pcpartspicker.com to verify), and the goal is to"
        "meet the technical requirements provided by the Chief Architect but"
        "minimize cost as much as possible."
    ),
    tools=[search_tool],
    allow_delegation=False,
)


# Research task
research_task = Task(
    description=(
        "Assemble a parts list for a {purpose} system at or under {budget}."
        "This system should meet the requirement provided by the Chief"
        "Architect, and the list should include source links and prices for"
        "each component required.  Do not forget to include cables for power"
        "and network in the budget requirements."
        "Please reference AI/ML blogs to determine the best hardware"
        "configurations as of 2024."
    ),
    expected_output="A comprehensive list of all parts, prices, and sources to build the system.",
    tools=[search_tool],
    agent=researcher,
)

# Writing task with language model configuration
write_task = Task(
    description=(
        "Find the cheapest price for a suitably sufficient part for a {purpose} system"
        "under {budget}. The existing parts in the list, when added to this"
        "part, should remain under the cost of {budget}.  Concatenate the "
        "source link, price, and item information to a new line in the file."
        "Make sure that this part is not included in the list if a sufficient"
        "part of combination of parts already exists to serve the same purpose"
        "in the system.  For example, if the list already contains 64GB or RAM"
        "then do not add any more RAM parts to the list.  All multiples of"
        "parts required to complete a system should be identical."
    ),
    expected_output="A 4 paragraph article on {purpose} advancements formatted as markdown.",
    tools=[search_tool],
    agent=writer,
    async_execution=False,
    output_file="part-list.md",  # Example of output customization
)


# Forming the tech-focused crew with enhanced configurations
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,  # Optional: Sequential task execution is default
)

# Starting the task execution process with enhanced feedback
result = crew.kickoff(
    inputs={"budget": 3000, "purpose": "Machine Learning workstation"}
)
print(result)
