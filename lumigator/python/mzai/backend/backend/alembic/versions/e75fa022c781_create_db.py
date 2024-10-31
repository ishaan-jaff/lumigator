"""Create DB

Revision ID: e75fa022c781
Revises:
Create Date: 2024-10-23 16:46:49.393141

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'e75fa022c781' # pragma: allowlist secret
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('datasets',
    sa.Column('filename', sa.String(), nullable=False),
    sa.Column(
        'format',
        sa.Enum('EXPERIMENT', name='datasetformat'),
        nullable=False
    ),
    sa.Column('size', sa.Integer(), nullable=False),
    sa.Column('ground_truth', sa.Boolean(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column(
        'created_at',
        sa.DateTime(timezone=True),
        server_default=sa.text('(CURRENT_TIMESTAMP)'),
        nullable=False
    ),
    sa.PrimaryKeyConstraint('id'),
    )
    op.create_table('jobs',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('description', sa.String(), nullable=False),
    sa.Column(
        'status',
        sa.Enum('CREATED', 'RUNNING', 'FAILED', 'SUCCEEDED', name='jobstatus'),
        nullable=False
    ),
    sa.Column(
        'created_at',
        sa.DateTime(timezone=True),
        server_default=sa.text('(CURRENT_TIMESTAMP)'),
        nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('job-results',
    sa.Column('job_id', sa.Uuid(), nullable=False),
    sa.Column('metrics', sa.JSON(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('job_id'),
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('job-results')
    op.drop_table('jobs')
    op.drop_table('datasets')
    # ### end Alembic commands ###