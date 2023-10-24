# Generated by Django 4.2.6 on 2023-10-22 07:34

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('learning_log', '0020_job'),
    ]

    operations = [
        migrations.CreateModel(
            name='Candidate_application',
            fields=[
                ('app_id', models.IntegerField(primary_key=True, serialize=False)),
                ('app_status', models.CharField(default=' ', max_length=50)),
                ('candidate_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='learning_log.candidate')),
                ('job_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='learning_log.job')),
            ],
        ),
    ]