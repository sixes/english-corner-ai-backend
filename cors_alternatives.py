# Alternative CORS configuration for debugging
# Use this if the specific domains don't work

# Replace the CORS middleware section in rag_backend.py with this:

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for debugging
    allow_credentials=False,  # Set to False when using "*"
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# OR if you want to be more specific:

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.englishcorner.cyou",
        "https://englishcorner.cyou", 
        "https://*.englishcorner.cyou",  # Wildcard subdomain
        "http://localhost:3000",
        "http://localhost:5173",
        "https://localhost:3000",
        "https://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["Accept", "Accept-Language", "Content-Language", "Content-Type", "Authorization"],
    expose_headers=["*"]
)
