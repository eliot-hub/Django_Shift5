# Generated by Django 4.2.6 on 2023-10-22 14:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('learning_log', '0023_alter_table1_photo'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='table1',
            name='photo',
        ),
        migrations.AddField(
            model_name='candidate',
            name='photo',
            field=models.ImageField(default=' ', upload_to='images/'),
        ),
    ]
