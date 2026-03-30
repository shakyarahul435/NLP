# AT82.05 NLU Assignment 7: MCP Server, AI Agent, and External Tool Integration

## Overview
This repository contains the setup and documentation for an integrated AI Agent ecosystem built using the Model Context Protocol (MCP) in n8n. The project demonstrates practical Natural Language Understanding (NLU) by using a Large Language Model (Groq) to manage real-world scheduling through Telegram and Google Calendar.

## Architecture and Features

### 1. MCP Infrastructure and Server Setup
- **Local deployment:** n8n is deployed locally using Docker and exposed to the internet via ngrok.
- **MCP server:** An n8n workflow acts as the MCP server, exposing internal tools such as Calculator and Date/Time Formatter to the agent.
- **AI agent client:** A secondary n8n workflow hosts the AI Agent. It uses:
  - Groq API for language processing
  - Window Buffer or Simple Memory for conversational context
  - MCP Client node to communicate with MCP server tools

### 2. External Integrations
- **Telegram Bot API:** The agent is connected to Telegram through a webhook/trigger so users can send natural language commands from Telegram.
- **Google Calendar API:** The agent is configured with Google Calendar tools to create, read, and manage events based on user prompts.

## Project Scheduling Automation
The agent can process a single natural language request and generate a full project schedule in Google Calendar. It automatically books these events:

1. **1st Phase:** Literature Review
2. **2nd Phase:** Project Proposal
3. **3rd Phase:** Update Progress
4. **4th Phase:** Final Presentation

## Prerequisites
- Docker and Docker Compose
- ngrok (or a similar tunneling service)
- Telegram Bot Token (from BotFather)
- Google Cloud Console credentials (Calendar API OAuth)
- Groq API key

## Setup Instructions

### 1. Start n8n
Run your local n8n instance with Docker Compose:

```bash
docker-compose up -d
```

### 2. Expose localhost
Start an ngrok tunnel to expose n8n publicly:

```bash
ngrok http 5678
```

### 3. Import workflows
Import the provided workflow JSON files into your n8n instance:
- MCP Server workflow
- AI Agent workflow

### 4. Configure credentials
Add credentials to the corresponding n8n nodes:
- Groq API key
- Telegram Bot token
- Google Calendar OAuth credentials

### 5. Activate workflows
Set both workflows to **Active**:
- MCP Server workflow
- AI Agent workflow

## Usage
Open Telegram, go to your bot, and send a message like:

> Please schedule my project phases in my Google Calendar.

The bot will interpret the request, create all four phase events in Google Calendar, and reply with a confirmation.

You can then ask follow-up questions such as:

> When is my Final Presentation?

## Acknowledgments
Completed for **AT82.05 Artificial Intelligence: Natural Language Understanding (NLU)**.

---
## Author
Rahul Shakya <br/>
st125982 <br/>
Asian Institute of Technology - AIT