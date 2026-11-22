from pathlib import Path
from decouple import config

# =========================
# BASE CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = config("SECRET_KEY", default="local-development-only-change-me")

def _read_debug(value):

    if isinstance(value, bool):
        return value

    if value is None:
        return False

    value = str(value).strip().lower()

    if value in {"1", "true", "yes", "on", "debug"}:
        return True

    if value in {"0", "false", "no", "off", "release", "prod", "production"}:
        return False

    return False


DEBUG = config(
    "DEBUG",
    default="True",
    cast=_read_debug
)

ALLOWED_HOSTS = config(
    "ALLOWED_HOSTS",
    default="localhost,127.0.0.1"
).split(",")


# =========================
# INSTALLED APPS
# =========================

INSTALLED_APPS = [
    # Django Default
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    # Third Party
    "rest_framework",
    "rest_framework_simplejwt",
    "corsheaders",
    "django_filters",
    "drf_yasg",

    # Local Apps
    "apps.accounts",
    "apps.roles_permissions.apps.RolesPermissionsConfig",
]


# =========================
# MIDDLEWARE
# =========================

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]


# =========================
# URLS
# =========================

ROOT_URLCONF = "config.urls"


# =========================
# TEMPLATES
# =========================

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]


# =========================
# WSGI
# =========================

WSGI_APPLICATION = "config.wsgi.application"


# =========================
# DATABASE
# =========================

if config("DB_NAME", default=""):
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": config("DB_NAME"),
            "USER": config("DB_USER", default=""),
            "PASSWORD": config("DB_PASSWORD", default=""),
            "HOST": config("DB_HOST", default="localhost"),
            "PORT": config("DB_PORT", default="5432"),
        }
    }
else:
    DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": BASE_DIR / "db.sqlite3"}}


# =========================
# PASSWORD VALIDATION
# =========================

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# =========================
# INTERNATIONALIZATION
# =========================

LANGUAGE_CODE = "en-us"

TIME_ZONE = "Asia/Ho_Chi_Minh"

USE_I18N = True

USE_TZ = True


# =========================
# STATIC FILES
# =========================

STATIC_URL = "static/"


# =========================
# DEFAULT PRIMARY KEY
# =========================

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# =========================
# DJANGO REST FRAMEWORK
# =========================

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "apps.accounts.authentication.AccountJWTAuthentication",
    ),
    "DEFAULT_FILTER_BACKENDS": (
        "django_filters.rest_framework.DjangoFilterBackend",
    ),
}

SIMPLE_JWT = {"USER_ID_FIELD": "user_id"}


# =========================
# CORS
# =========================

CORS_ALLOWED_ORIGINS = config(
    "CORS_ALLOWED_ORIGINS", default="http://localhost:8081,http://127.0.0.1:8081"
).split(",")

EMAIL_BACKEND = config("EMAIL_BACKEND", default="django.core.mail.backends.console.EmailBackend")

EMAIL_HOST = config("EMAIL_HOST", default="localhost")
EMAIL_PORT = config("EMAIL_PORT", default=25, cast=int)
EMAIL_HOST_USER = config("EMAIL_HOST_USER", default="")
EMAIL_HOST_PASSWORD = config("EMAIL_HOST_PASSWORD", default="")
EMAIL_USE_TLS = config("EMAIL_USE_TLS", default=False, cast=bool)
DEFAULT_FROM_EMAIL = config("DEFAULT_FROM_EMAIL", default="no-reply@localhost")

VERIFY_CODE_EXPIRE_MINUTES = config(
    "VERIFY_CODE_EXPIRE_MINUTES",
    default=10,
    cast=int,
)
