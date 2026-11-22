"""Create an admin for the standalone Account model without a default password."""

from getpass import getpass

from django.contrib.auth.hashers import make_password
from django.core.management.base import BaseCommand, CommandError
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from rest_framework.exceptions import ValidationError as RestValidationError

from apps.accounts.models import Account
from apps.accounts.validators import validate_password
from apps.roles_permissions.models import RoleAccount


class Command(BaseCommand):
    help = "Create an ADMIN account with an interactively entered password."

    def add_arguments(self, parser):
        parser.add_argument("email")

    def handle(self, *args, **options):
        email = options["email"].strip().lower()
        try:
            validate_email(email)
        except ValidationError as exc:
            raise CommandError("Invalid email address") from exc
        if Account.objects.filter(email=email).exists():
            raise CommandError("Account already exists")

        password = getpass("Password: ")
        if password != getpass("Confirm password: "):
            raise CommandError("Passwords do not match")
        try:
            validate_password(password)
        except RestValidationError as exc:
            raise CommandError(str(exc.detail)) from exc

        role = RoleAccount.objects.get(role_code=RoleAccount.RoleCode.ADMIN)
        Account.objects.create(email=email, password=make_password(password), role=role)
        self.stdout.write(self.style.SUCCESS(f"Created ADMIN account: {email}"))
