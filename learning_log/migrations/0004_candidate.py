# Generated by Django 4.2.6 on 2023-10-21 16:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('learning_log', '0003_delete_table2'),
    ]

    operations = [
        migrations.CreateModel(
            name='Candidate',
            fields=[
                ('candidate_id', models.IntegerField(primary_key=True, serialize=False)),
            ],
        ),
    ]