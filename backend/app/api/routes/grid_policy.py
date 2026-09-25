"""API Routes for Variable-Resolution Grid Policy."""

from fastapi import APIRouter, HTTPException, status
from ...models.map_schemas import GridPolicyConfig
from ...services.resolution_policy import ResolutionPolicyService

router = APIRouter(tags=["Grid Policy"])


@router.get(
    "/grid-policy",
    response_model=GridPolicyConfig,
    summary="Get Variable-Resolution Grid Policy",
    description="Retrieve the active distance thresholds, base resolutions, and refinement overrides.",
)
async def get_grid_policy() -> GridPolicyConfig:
    return ResolutionPolicyService.get_policy()


@router.put(
    "/grid-policy",
    response_model=GridPolicyConfig,
    summary="Update Variable-Resolution Grid Policy",
    description=(
        "Update and persist grid foveation parameters and safety override flags. "
        "Resolution sizes are normalised to a strict 2:1 hierarchy so the parent/child "
        "cell-key arithmetic stays spatially aligned."
    ),
)
async def update_grid_policy(new_policy: GridPolicyConfig) -> GridPolicyConfig:
    ok, msg = ResolutionPolicyService.validate_policy(new_policy)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=msg or "Invalid policy configuration.",
        )

    normalised = ResolutionPolicyService.normalise_policy(new_policy)
    try:
        updated = ResolutionPolicyService.update_policy(normalised)
        return updated
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update grid policy: {str(e)}",
        )
