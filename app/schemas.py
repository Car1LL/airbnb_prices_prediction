from pydantic import BaseModel


class ListingInput(BaseModel):
    id: int
    property_type: str
    room_type: str
    amenities: str
    accommodates: int
    bathrooms: float | None = None
    bed_type: str
    cancellation_policy: str
    cleaning_fee: bool
    city: str
    description: str | None = None
    first_review: str | None = None
    host_has_profile_pic: str | None = None
    host_identity_verified: str | None = None
    host_response_rate: str | None = None
    host_since: str | None = None
    instant_bookable: str
    last_review: str | None = None
    latitude: float
    longitude: float
    name: str
    neighbourhood: str | None = None
    number_of_reviews: int
    review_scores_rating: float | None = None
    thumbnail_url: str | None = None
    zipcode: str | None = None
    bedrooms: float | None = None
    beds: float | None = None