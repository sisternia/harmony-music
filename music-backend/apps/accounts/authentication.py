"""Authenticate the Account model used by the accounts API."""

from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import AuthenticationFailed, InvalidToken
from rest_framework_simplejwt.settings import api_settings

from .models import Account


class AccountJWTAuthentication(JWTAuthentication):
    def get_user(self, validated_token):
        try:
            user_id = validated_token[api_settings.USER_ID_CLAIM]
        except KeyError as exc:
            raise InvalidToken("Token contained no account ID") from exc

        try:
            return Account.objects.select_related("role").get(user_id=user_id)
        except Account.DoesNotExist as exc:
            raise AuthenticationFailed("Account not found", code="user_not_found") from exc
