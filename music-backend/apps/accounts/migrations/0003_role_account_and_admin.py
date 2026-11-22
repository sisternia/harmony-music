# Generated manually to link accounts to roles.

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0002_verifyaccount"),
        ("roles_permissions", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="account",
            name="role",
            field=models.ForeignKey(
                db_column="role_id",
                default=2,
                on_delete=django.db.models.deletion.PROTECT,
                related_name="accounts",
                to="roles_permissions.roleaccount",
            ),
        ),
    ]
