# Generated by Django 4.2.6 on 2023-10-20 07:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('learning_log', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Table2',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('std', models.CharField(max_length=20)),
            ],
        ),
    ]
