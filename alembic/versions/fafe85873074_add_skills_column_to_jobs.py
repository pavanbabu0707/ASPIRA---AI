"""add skills column to jobs

Revision ID: fafe85873074
Revises: 6ba9bb494894
Create Date: 2025-11-04 16:31:26.629176

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

def upgrade():
    op.add_column('jobs', sa.Text(name='skills'))

def downgrade():
    op.drop_column('jobs', 'skills')


# revision identifiers, used by Alembic.
revision: str = 'fafe85873074'
down_revision: Union[str, Sequence[str], None] = '6ba9bb494894'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
