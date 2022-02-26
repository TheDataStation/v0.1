from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from ..models.policy import Policy
from ..schemas.policy import PolicyCreate

def create_policy(db: Session, policy: PolicyCreate):
    db_policy = Policy(user_id=policy.user_id,
                       api=policy.api,
                       data_id=policy.data_id,)
    try:
        db.add(db_policy)
        # print("add")
        db.commit()
        # print("commit")
        db.refresh(db_policy)
        # print("refresh")
    except SQLAlchemyError as e:
        db.rollback()
        return None
    return db_policy

def remove_policy(db: Session, policy: PolicyCreate):
    try:
        db.query(Policy).filter(Policy.user_id == policy.user_id,
                                Policy.api == policy.api,
                                Policy.data_id == policy.data_id).delete()
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        return None
    return "success"

def get_all_policies(db: Session):
    policies = db.query(Policy).all()
    return policies

def get_policy_for_user(db: Session, user_id: str):
    policies = db.query(Policy).filter(Policy.user_id == user_id).all()
    return policies

# The following function recovers the policy table from a list of Policy
def bulk_upload_policies(db: Session, policies):
    policies_to_add = []
    for policy in policies:
        cur_policy = Policy(user_id=policy.user_id,
                            api=policy.api,
                            data_id=policy.data_id,)
        policies_to_add.append(cur_policy)
    try:
        db.add_all(policies_to_add)
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        return None

    return "success"
