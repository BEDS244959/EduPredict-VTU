from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

from app.config import settings
from app.database import Base, engine
from app.routes import admin, auth, common, staff, students
from app.seed import seed_if_needed

app = FastAPI(title="EduPredict AI", version="1.0.0", docs_url="/api/docs", redoc_url="/api/redoc")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exc(_request: Request, exc: HTTPException):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


@app.exception_handler(IntegrityError)
async def integrity(_request: Request, _exc: IntegrityError):
    return JSONResponse({"detail": "This record conflicts with existing data."}, status_code=400)


@app.exception_handler(Exception)
async def unhandled(_request: Request, exc: Exception):
    return JSONResponse({"detail": "Something went wrong. Please try again."}, status_code=500)


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
    seed_if_needed()


@app.get("/api/health")
def health():
    return {"ok": True, "name": settings.APP_NAME}


app.include_router(auth.router, prefix="/api")
app.include_router(students.router, prefix="/api")
app.include_router(staff.router, prefix="/api")
app.include_router(admin.router, prefix="/api")
app.include_router(common.router, prefix="/api")
