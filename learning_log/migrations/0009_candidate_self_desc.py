# Generated by Django 4.2.6 on 2023-10-21 16:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('learning_log', '0008_candidate_city_candidate_country_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='candidate',
            name='self_desc',
            field=models.TextField(default=' ', max_length=255),
        ),
    ]
