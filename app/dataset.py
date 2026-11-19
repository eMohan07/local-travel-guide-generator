"""Dataset preparation module for Zephyria local travel guide generator.

Contains curated travel data for the fictional city of Zephyria,
and formats travel interests and recommendations into input-output pairs
suitable for fine-tuning language models with LoRA.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

CITY_NAME = "Zephyria"
CITY_DESCRIPTION = (
    "Zephyria is a breathtaking fictional coastal city where antique clockwork towers, "
    "bioluminescent gardens, lantern-lit canal districts, and vibrant culinary alleys meet the sea."
)

# Curated dataset of travel destinations, interests, traveler profiles, and insider recommendations
TRAVEL_RECOMMENDATIONS_DATA: List[Dict[str, Any]] = [
    {
        "id": "zeph-001",
        "city": CITY_NAME,
        "district": "Old Quay & Seafront",
        "interests": ["Maritime History", "Fresh Seafood", "Coastal Walks"],
        "traveler_profile": "History and culinary enthusiast seeking authentic coastal maritime culture",
        "highlights": [
            "Salt-Breeze Promenade for panoramic views of historic galleon docks",
            "The Barnacle & Bell tavern for smoked chowder and freshly shucked sea-clams",
            "Zephyria Maritime Museum housed inside an ancient wooden dry-dock",
        ],
        "secret_spot": "The Whispering Breakwater at low tide, where harbor bells echo through sea caves.",
        "best_time": "Early morning or late afternoon for gentle sea breezes and golden hour sunlight.",
        "local_tip": "Ask the fishermen at Pier 4 for the catch-of-the-tide; they grill it on cedar planks right by the water.",
    },
    {
        "id": "zeph-002",
        "city": CITY_NAME,
        "district": "Lantern Quarter",
        "interests": ["Street Food", "Night Markets", "Live Acoustic Music"],
        "traveler_profile": "Night owl foodie looking for lively atmosphere, nocturnal bites, and indie tunes",
        "highlights": [
            "Canal-side Night Bazaar under thousands of floating crimson and amber silk lanterns",
            "Dragonfly Dumpling Den for pan-seared spiced plum dumplings",
            "Acoustic Alleyway where buskers and lute players perform in cozy cobblestone alcoves",
        ],
        "secret_spot": "The Gilded Kettle — an unmarked basement tea parlor serving spiced lotus chai until 2 AM.",
        "best_time": "8:00 PM to midnight when the lanterns reflect fully upon the mirror-still canals.",
        "local_tip": "Carry copper tokens for the street food stalls; stallholders give a 10% discount for exact change.",
    },
    {
        "id": "zeph-003",
        "city": CITY_NAME,
        "district": "Clockwork Ridge",
        "interests": ["Vintage Horology", "Neo-Victorian Architecture", "Panoramic Views"],
        "traveler_profile": "Architectural admirer and steampunk enthusiast interested in mechanical history",
        "highlights": [
            "The Grand Orrery Tower displaying a three-story astronomical gear mechanism",
            "Cobblestone gear-maker workshops along Pendulum Way",
            "The High Vista Cable Car traversing the gorge between upper and lower ridge",
        ],
        "secret_spot": "The Balcony of the 12th Gear: a hidden perch behind the town hall clock face overlooking the skyline.",
        "best_time": "Noon sharp to witness all 50 clock towers chime in harmonic resonance.",
        "local_tip": "Visit Master Thaddeus's workshop for brass souvenir compasses calibrated to Zephyria's magnetic ley lines.",
    },
    {
        "id": "zeph-004",
        "city": CITY_NAME,
        "district": "Mistwood Gardens",
        "interests": ["Bioluminescent Flora", "Nature Hiking", "Botanical Photography"],
        "traveler_profile": "Nature photographer and hiker eager to explore serene, otherworldly flora",
        "highlights": [
            "The Glow-Cap Fern Trail featuring natural bioluminescent fungi",
            "Emerald Conservatory housing endangered glass-petal orchids",
            "Whispering Willow Falls with moss-covered stepping stones",
        ],
        "secret_spot": "Luminescent Hollow: a hidden valley that glows in soft turquoise after twilight.",
        "best_time": "Dusk (around 7:30 PM) as daylight fades and the radiant flora begins to glow.",
        "local_tip": "Bring a tripod and avoid camera flash to preserve the delicate glow and natural ambience.",
    },
    {
        "id": "zeph-005",
        "city": CITY_NAME,
        "district": "Sunstone Bazaar",
        "interests": ["Handcrafted Ceramics", "Antique Textiles", "Spice Tasting"],
        "traveler_profile": "Art collector and sensory traveler looking for artisan crafts and exotic flavors",
        "highlights": [
            "The Glassblower's Court with live open-flame glass sculpting",
            "Saffron & Silk Arcade for hand-dyed indigo tapestries and rare mountain peppercorns",
            "The Clay Potters' Guild showcasing glazed terracotta tea sets",
        ],
        "secret_spot": "The Courtyard of Scent: an inner cloister offering customized cardamom perfume oils.",
        "best_time": "Mid-morning (10:00 AM) when fresh spice deliveries arrive and artisans open their studios.",
        "local_tip": "Polite bargaining is welcomed; ask the merchant the story behind each craft piece for a cordial discount.",
    },
    {
        "id": "zeph-006",
        "city": CITY_NAME,
        "district": "Crystal Cove",
        "interests": ["Sea Kayaking", "Coastal Caves", "Tidal Marine Life"],
        "traveler_profile": "Active outdoor adventurer wanting water recreation and scenic marine discovery",
        "highlights": [
            "Guided glass-bottom kayak excursions into Sapphire Sea Cave",
            "Tidal pool exploration along the Quartz Reef",
            "Sunset cliff trail with views of frolicking silver sea-otters",
        ],
        "secret_spot": "Mermaid's Grotto — accessible only by paddleboard at low tide through a narrow rock arch.",
        "best_time": "Low tide mornings for calm waters and optimal visibility into tidal marine pools.",
        "local_tip": "Rent neoprene booties from Cove Outfitters to protect feet against sharp quartz formations.",
    },
    {
        "id": "zeph-007",
        "city": CITY_NAME,
        "district": "The Scholar's Atoll",
        "interests": ["Rare Books", "Specialty Coffee", "Quiet Writing Spots"],
        "traveler_profile": "Literary nomad and remote writer seeking intellectual ambiance and fine brew",
        "highlights": [
            "The Floating Folio — a refurbished three-deck barge stocked with antique cartography and first editions",
            "Roasted Ink Cafe, known for cold-brew infused with vanilla bean and roasted chicory",
            "The Parchment Courtyard with outdoor reading hammocks beneath wisteria pergolas",
        ],
        "secret_spot": "The Bindery Solarium on the third floor of the Archivist Guild, bathed in warm, quiet daylight.",
        "best_time": "Late afternoon on weekdays for undisturbed quietude and golden sun on bookshelf aisles.",
        "local_tip": "Purchase a handmade leather bookmark stamped with the Atoll's wax seal as a timeless memento.",
    },
    {
        "id": "zeph-008",
        "city": CITY_NAME,
        "district": "Elysian Terraces",
        "interests": ["Sunset Wine Tastings", "Fine Dining", "Classical Quartets"],
        "traveler_profile": "Luxury traveler seeking refined gastronomy, panoramic terrace views, and chamber music",
        "highlights": [
            "Terrace Hill Vineyards offering flights of coastal amber wine",
            "Restaurant L'Aura perched on a cliff edge with seasonal 7-course tasting menus",
            "The Marble Amphitheatre hosting open-air twilight string quartet recitals",
        ],
        "secret_spot": "The Sommelier's Cellar Cleft: a candlelit underground basalt cave storing vintage vintages.",
        "best_time": "6:30 PM onwards to pair the sunset over the bay with an aperitif.",
        "local_tip": "Smart-casual dress code is customary; reserve terrace seating at least 48 hours in advance.",
    },
    {
        "id": "zeph-009",
        "city": CITY_NAME,
        "district": "Breeze Heights",
        "interests": ["Family-Friendly Fun", "Kite Flying", "Whimsical Puppet Shows"],
        "traveler_profile": "Family with children seeking cheerful outdoor activities, open green spaces, and theatre",
        "highlights": [
            "The Windmill Meadows with vibrant giant kite rentals and steady sea currents",
            "Marionette Square with hourly interactive children's folk tale performances",
            "The Honeycomb Bakery serving bee-shaped sweet cinnamon brioches",
        ],
        "secret_spot": "The Echoing Carousel: an antique carved wooden carousel where animals chime gentle melodies.",
        "best_time": "Sunny weekends between 11:00 AM and 3:00 PM when family workshops are in full swing.",
        "local_tip": "Pack a blanket for a lawn picnic; the bakery rents complete wicker picnic hampers.",
    },
    {
        "id": "zeph-010",
        "city": CITY_NAME,
        "district": "Shadowfall Caverns",
        "interests": ["Geothermal Mineral Baths", "Underground Jazz", "Speakeasies"],
        "traveler_profile": "Atmospheric explorer craving thermal relaxation followed by underground nightlife",
        "highlights": [
            "The Subterranean Thermal Baths naturally heated by volcanic limestone veins",
            "The Stalactite Lounge: an acoustic underground jazz club carved into cavern rock",
            "The Mineral Vapor Rooms infused with wild mountain eucalyptus",
        ],
        "secret_spot": "The Grotto of Echoes: a quiet warm pool behind the waterfall inside Cavern Three.",
        "best_time": "Late evening (after 9:00 PM) when live saxophone sets commence in the Stalactite Lounge.",
        "local_tip": "Bring swimwear and a warm robe; the caverns stay a cozy 28°C year-round.",
    },
    {
        "id": "zeph-011",
        "city": CITY_NAME,
        "district": "Starlight Pier",
        "interests": ["Stargazing", "Night Astronomy", "Oceanfront Dessert Bars"],
        "traveler_profile": "Romantic couple interested in astronomy, celestial views, and gourmet sweets",
        "highlights": [
            "The Pier Observatory equipped with brass telescopes pointing toward coastal constellations",
            "Moonstone Patisserie specializing in lavender macarons and caramelized moon-tarts",
            "Luminescent Boardwalk featuring glowing embedded sea-glass pavers",
        ],
        "secret_spot": "The Telescope Pavilion at the pier's far tip, where dark-sky conditions reveal the Milky Way.",
        "best_time": "Clear moonless nights between 10:00 PM and midnight.",
        "local_tip": "The local astronomical society hosts free public sky-tours on Wednesday and Saturday nights.",
    },
    {
        "id": "zeph-012",
        "city": CITY_NAME,
        "district": "Copperworks District",
        "interests": ["Steampunk Art", "Artisan Metalcraft", "Craft Microbreweries"],
        "traveler_profile": "Creative maker and craft beer lover intrigued by industrial aesthetic and ironwork",
        "highlights": [
            "The Forge & Flagon brewery with copper brewing kettles and smoked malt ales",
            "The Blacksmith's Arcade featuring custom hammered kitchenware and jewelry",
            "Foundry Sculpture Park displaying kinetic bronze sculptures powered by water wheels",
        ],
        "secret_spot": "The Anvil Alley Speakeasy: enter through the tool cabinet behind the blacksmith stall.",
        "best_time": "Friday late afternoons for happy hour tastings and live forge demonstrations.",
        "local_tip": "Order the 'Smoked Amber Porter' sampler; it comes served on a miniature cast-iron skillet.",
    },
    {
        "id": "zeph-013",
        "city": CITY_NAME,
        "district": "Whispering Orchards",
        "interests": ["Agritourism", "Cider Pressing", "Farm-to-Table Gastronomy"],
        "traveler_profile": "Eco-conscious traveler who appreciates organic agriculture, orchard walks, and seasonal produce",
        "highlights": [
            "Sun-blush Pear Orchards offering self-pick seasonal fruit baskets",
            "The Old Mill Cider House with fresh barrel-tapped sparkling apple-cider",
            "Barnwood Hearth Bistro serving wood-fired flatbreads with local goat cheese and figs",
        ],
        "secret_spot": "The Blossom Hammocks suspended in the century-old heirloom quince grove.",
        "best_time": "Autumn harvest or spring blossom season, preferably around 1:00 PM for lunch.",
        "local_tip": "Book the tractor shuttle from Clockwork Ridge for a scenic ride through the rolling orchard hills.",
    },
    {
        "id": "zeph-014",
        "city": CITY_NAME,
        "district": "Azure Citadel",
        "interests": ["Medieval Folklore", "Sword Craft", "Historical Reenactments"],
        "traveler_profile": "History buff fascinated by knightly traditions, defensive fortresses, and ancient myths",
        "highlights": [
            "The Rampart Walkway with 360-degree views over Zephyria Bay and fortress bastions",
            "The Armourer’s Hall displaying ceremonial shields and historic bladed relics",
            "The Tournament Courtyard featuring daily choreographed historical fencing demonstrations",
        ],
        "secret_spot": "The Whispering Well in the lower bailey, said to grant wishes for safe voyages.",
        "best_time": "2:00 PM to catch the Citadel Guard change and the herald trumpet salute.",
        "local_tip": "Audio guides are narrated by the City Historian and reveal secret smuggling tunnels under the battlements.",
    },
    {
        "id": "zeph-015",
        "city": CITY_NAME,
        "district": "Windward Cliffs",
        "interests": ["Paragliding", "Extreme Coastal Trails", "Sunset Drone Photography"],
        "traveler_profile": "Adrenaline seeker and aerial photographer eager for dramatic landscapes and windswept heights",
        "highlights": [
            "Falcon Point Paragliding Launch for tandem flights soaring over coastal surf",
            "The Razorback Ridge Trail featuring sheer cliffs and dramatic wave crashes",
            "Seagull Lookout Point equipped with wind sensors and panoramic camera perches",
        ],
        "secret_spot": "The Hawk’s Eyrie: an isolated stone shelter jutting out 200 meters above the crashing waves.",
        "best_time": "Late afternoon 1 hour before sunset when thermals are steady and lighting is spectacular.",
        "local_tip": "Wind conditions shift rapidly; always register your flight or trail route at the Ranger Outpost.",
    },
    {
        "id": "zeph-016",
        "city": CITY_NAME,
        "district": "Silk & Spindle Alley",
        "interests": ["High Fashion", "Bespoke Tailoring", "Silversmithing"],
        "traveler_profile": "Fashion-conscious explorer looking for custom garments, jewelry design, and styling",
        "highlights": [
            "Atelier Zephyr offering bespoke linen coats tailored in 24 hours",
            "The Silver Filigree Guild crafting delicate sea-shell inspired silver jewelry",
            "Textile Dyeing Vats showcasing traditional saffron, madder root, and indigo dyes",
        ],
        "secret_spot": "The Ribbon Garden: a charming courtyard draped with drying silk ribbons fluttering in the wind.",
        "best_time": "11:00 AM when natural light fills the tailor fitting rooms.",
        "local_tip": "Bring your favorite fabric swatch; artisans can customize embroidery patterns within a single afternoon.",
    },
    {
        "id": "zeph-017",
        "city": CITY_NAME,
        "district": "The Alchemist's Quarter",
        "interests": ["Herbal Tea Blends", "Botanical Aromatherapy", "Holistic Wellness"],
        "traveler_profile": "Wellness traveler looking to rejuvenate through medicinal herbs, essential oils, and quiet tea rooms",
        "highlights": [
            "Apothecary 99 blending personalized loose-leaf teas based on your temperament",
            "The Essential Garden with steam-distilled rosemary, cedar, and lavender oils",
            "The Sunlit Salt Room offering respiratory halotherapy sessions",
        ],
        "secret_spot": "The Sage & Honey Sanctuary: an inner garden dedicated to silent meditation and herbal infusions.",
        "best_time": "Early afternoon (2:00 PM) for tranquil post-lunch relaxation and tea consultations.",
        "local_tip": "Sample the 'Zephyrian Dream' blend — a mix of blue chamomile, lavender, and mountain mint for deep rest.",
    },
    {
        "id": "zeph-018",
        "city": CITY_NAME,
        "district": "Moonlit Lagoon",
        "interests": ["Stand-up Paddleboarding", "Bioluminescent Swims", "Campfire Acoustic Sessions"],
        "traveler_profile": "Water sports enthusiast and romantic wanderer seeking serene nighttime aquatic adventures",
        "highlights": [
            "Guided paddleboarding across neon-blue bioluminescent waters",
            "Lagoon Beach bonfire circle with toasted marshmallows and folk acoustic guitars",
            "Floating raft bar serving chilled coconut-lime spritzers",
        ],
        "secret_spot": "The Driftwood Archway: a secluded lagoon inlet where bioluminescence is brightest.",
        "best_time": "9:30 PM on new moon nights for maximum bioluminescent glow against dark water.",
        "local_tip": "Waterproof pouches for phones and cameras are provided for free by the lagoon paddle shack.",
    },
    {
        "id": "zeph-019",
        "city": CITY_NAME,
        "district": "Cobblestone Commons",
        "interests": ["Street Performance", "Busking Culture", "Open-air Chess"],
        "traveler_profile": "Casual city stroller who loves people-watching, impromptu theatrics, and friendly games",
        "highlights": [
            "The Grand Chess Pavilion with life-sized marble chessboards and active challenge matches",
            "Fountain of Minstrels featuring rotating juggling and acrobatics acts",
            "Cafe de la Place with curbside wicker chairs and buttered brioches",
        ],
        "secret_spot": "The Poet’s Nook beneath the weeping elm tree, where local writers pin handwritten poetry.",
        "best_time": "3:00 PM to 6:00 PM when the square fills with lively artists and chess grandmasters.",
        "local_tip": "Anyone can challenge the 'Table 1 Master'; if you survive 20 moves, they buy your coffee!",
    },
    {
        "id": "zeph-020",
        "city": CITY_NAME,
        "district": "Skyward Conservatory",
        "interests": ["High-altitude Orchids", "Glasshouse Architecture", "Afternoon High Tea"],
        "traveler_profile": "Sophisticated sightseer looking for botanical elegance, fine teas, and breathtaking panoramic glass dome architecture",
        "highlights": [
            "The Cloud Dome: a five-tier Victorian geodesic glasshouse suspended over the city cliff",
            "The Royal Orchid Atrium displaying over 400 airborne orchid varieties",
            "Skyline Tea Salon serving three-tier pastries, scones with clotted cream, and floral teas",
        ],
        "secret_spot": "The Palm Frond Catwalk: an elevated metal walkway directly beneath the glass spire.",
        "best_time": "4:00 PM for the traditional afternoon tea sitting as late sun floods the glass dome.",
        "local_tip": "Try the signature 'Cloud Mist Scone' infused with rosewater and candied orange peel.",
    },
]


def format_instruction_prompt(
    destination: str,
    interests: List[str] | str,
    traveler_profile: Optional[str] = None,
    city: str = CITY_NAME,
) -> str:
    """Format the input prompt for LLM instruction fine-tuning."""
    if isinstance(interests, list):
        interests_str = ", ".join(interests)
    else:
        interests_str = str(interests)

    prompt = (
        f"Below is an instruction that describes a task, paired with an input that provides further context. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n"
        f"You are an expert travel concierge for the fictional city of {city}. "
        f"Generate a personalized, appealing local travel guide and recommendations based on the tourist's destination and interests.\n\n"
        f"### Input:\n"
        f"City: {city}\n"
        f"Destination District: {destination}\n"
        f"Travel Interests: {interests_str}\n"
    )
    if traveler_profile:
        prompt += f"Traveler Profile: {traveler_profile}\n"

    prompt += "\n### Response:\n"
    return prompt


def format_recommendation_response(
    district: str,
    highlights: List[str],
    secret_spot: str,
    best_time: str,
    local_tip: str,
    city: str = CITY_NAME,
) -> str:
    """Format the output recommendation guide for language model training."""
    highlights_bulleted = "\n".join(f"- {h}" for h in highlights)

    response = (
        f"Welcome to {city}'s {district}! Here is your personalized local guide:\n\n"
        f"🌟 **Top Highlights & Curated Experiences:**\n"
        f"{highlights_bulleted}\n\n"
        f"🤫 **Local Secret / Hidden Gem:**\n"
        f"{secret_spot}\n\n"
        f"⏰ **Best Time to Visit:**\n"
        f"{best_time}\n\n"
        f"💡 **Insider Tip:**\n"
        f"{local_tip}"
    )
    return response


def format_dataset_item(raw_item: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a raw travel recommendation item into a structured input-output training pair."""
    prompt = format_instruction_prompt(
        destination=raw_item["district"],
        interests=raw_item["interests"],
        traveler_profile=raw_item.get("traveler_profile"),
        city=raw_item.get("city", CITY_NAME),
    )
    response = format_recommendation_response(
        district=raw_item["district"],
        highlights=raw_item["highlights"],
        secret_spot=raw_item["secret_spot"],
        best_time=raw_item["best_time"],
        local_tip=raw_item["local_tip"],
        city=raw_item.get("city", CITY_NAME),
    )

    # Full text for causal language modeling
    full_text = f"{prompt}{response}"

    return {
        "id": raw_item["id"],
        "city": raw_item.get("city", CITY_NAME),
        "district": raw_item["district"],
        "interests": raw_item["interests"],
        "traveler_profile": raw_item.get("traveler_profile", ""),
        "input": prompt,
        "output": response,
        "text": full_text,
    }


def prepare_travel_dataset(
    output_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """Build the dataset of input-output pairs and optionally save to disk."""
    dataset = [format_dataset_item(item) for item in TRAVEL_RECOMMENDATIONS_DATA]

    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

    return dataset
