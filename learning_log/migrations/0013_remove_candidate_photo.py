# Generated by Django 4.2.6 on 2023-10-21 21:24

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('learning_log', '0012_alter_candidate_disability_alter_candidate_email_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='candidate',
            name='photo',
        ),
    ]
