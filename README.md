Input Image 1 (100×100×3)    Input Image 2 (100×100×3)
        │                           │
        ▼                           ▼
┌─────────────────┐       ┌─────────────────┐
│                 │       │                 │
│  Embedding      │       │  Embedding      │
│  Network        │       │  Network        │
│  (Shared        │       │  (Shared        │
│   Weights)      │       │   Weights)      │
│                 │       │                 │
└─────────────────┘       └─────────────────┘
        │                           │
        ▼                           ▼
   Embedding 1                 Embedding 2
   (4096-dim)                  (4096-dim)
        │                           │
        └────────────┬─────────────┘
                     ▼
             ┌───────────────┐
             │ L1 Distance   │
             │ Layer         │
             └───────────────┘
                     │
                     ▼
             ┌───────────────┐
             │ Dense Layer   │
             │ (1 unit,      │
             │  sigmoid)     │
             └───────────────┘
                     │
                     ▼
             Similarity Score
                  (0-1)
