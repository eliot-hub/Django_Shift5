# Generated by Django 4.2.6 on 2023-10-22 14:12

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('learning_log', '0024_remove_table1_photo_candidate_photo'),
    ]

    operations = [
        migrations.AlterField(
            model_name='candidate',
            name='photo',
            field=models.ImageField(default=' ', upload_to='photos/'),
        ),
        migrations.CreateModel(
            name='Candidate_Docs',
            fields=[
                ('doc_id', models.IntegerField(primary_key=True, serialize=False)),
                ('doc_name', models.CharField(default=' ', max_length=50)),
                ('doc_data', models.FileField(default=' ', upload_to='resumes/')),
                ('candidate_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='learning_log.candidate')),
            ],
        ),
    ]
