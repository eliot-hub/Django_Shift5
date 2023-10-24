# Generated by Django 4.2.6 on 2023-10-22 07:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('learning_log', '0019_candidate_reference'),
    ]

    operations = [
        migrations.CreateModel(
            name='job',
            fields=[
                ('job_id', models.IntegerField(primary_key=True, serialize=False)),
                ('job_title', models.CharField(default=' ', max_length=50)),
                ('job_position', models.CharField(default=' ', max_length=50)),
                ('company', models.CharField(default=' ', max_length=50)),
                ('job_location', models.CharField(default=' ', max_length=50)),
                ('job_type', models.CharField(default=' ', max_length=50)),
                ('job_desc', models.TextField(default=' ', max_length=100)),
                ('job_role', models.CharField(default=' ', max_length=50)),
                ('job_options', models.CharField(default=' ', max_length=50)),
            ],
        ),
    ]
