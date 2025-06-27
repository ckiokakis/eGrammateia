# eGrammateia

We present one of the first fully Greek-language student-administrative support systems that combines a Retrieval-Augmented Generation (RAG) pipeline with an open-source Large Language Model (LLM).

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have Python 3.10+ installed. You can check by running:

  ```bash
  python --version
  ```
* You have [pip](https://pip.pypa.io/en/stable/) installed for Python package management.
* You have [Node.js](https://nodejs.org/) (v14+) and npm installed. You can check by running:

  ```bash
  node --version
  npm --version
  ```

## Project Structure

```plain
eGrammateia/
├── backend/
│   ├── app.py
│   └── requirements.txt
└── frontend/
    ├── package.json
    └── src/
```

## Deployment

Follow these steps to deploy both the backend and frontend of the application.

### 1. Backend

1. Navigate to the backend directory:

   ```bash
   cd backend
   ```
2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```
3. Start the backend application:

   ```bash
   python app.py
   ```

The backend should now be running on its default port (`http://localhost:8090`).

### 2. Frontend

1. Open a new terminal window and navigate to the frontend directory:

   ```bash
   cd frontend
   ```
2. Install the required npm packages:

   ```bash
   npm install
   ```
3. Start the frontend development server:

   ```bash
   npm run start
   ```

The frontend should now be running at `http://localhost:3000` and will proxy API requests to the backend.
