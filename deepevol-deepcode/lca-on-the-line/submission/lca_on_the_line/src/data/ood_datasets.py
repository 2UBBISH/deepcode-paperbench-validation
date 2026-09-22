"""Out-of-distribution ImageNet dataset loaders for the LCA-on-the-Line benchmark.

The paper (§4 "Dataset Setup") evaluates five severe natural distribution-shift
datasets on top of ImageNet-1k:

    * ImageNet-v2   (Recht et al., 2019)   -- MatchedFrequency variant only
    * ImageNet-S    (Wang et al., 2019)    -- "ImageNet-Sketch"
    * ImageNet-R    (Hendrycks et al., 2021) -- "ImageNet-Rendition"
    * ImageNet-A    (Hendrycks et al., 2021) -- "ImageNet-Adversarial"
    * ObjectNet     (Barbu et al., 2019)

Sources (Addendum, "ImageNet datasets"):

    - ImageNet-v2: https://imagenetv2.org/ , and the paper uses the
      ``MatchedFrequency`` split from commit ``d626240`` of
      https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main
    - ImageNet-S: https://huggingface.co/datasets/songweig/imagenet_sketch
    - ImageNet-R: https://github.com/hendrycks/imagenet-r
    - ImageNet-A: https://github.com/hendrycks/natural-adv-examples
    - ObjectNet: https://objectnet.dev/

Every loader returns an :class:`~src.data.imagenet.ImageNetIDDataset`-style
``torch.utils.data.Dataset`` whose labels are remapped into the canonical
1000-class ImageNet (WordNet) index ordering, so that the logits of all 75
models line up with ``wordnet.py`` class ids and the LCA matrix.

The OOD variants do not all share ImageNet's label ordering:

    * ImageNet-A / ImageNet-R ship ``wnids.txt`` (or folder names with synset
      ids) -> remap by *wnid*.
    * ObjectNet ships its own 313-class folder names plus a mapping to the
      1000 ImageNet classes -> remap by *name*.
    * ImageNet-v2 (HF) / ImageNet-S (HF) already use ImageNet's label order.

Transform notes: the paper requires ID/OOD preprocessing to be identical, so
:func:`src.data.imagenet.build_transform` is reused here unchanged.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .imagenet import (  # noqa: F401  (re-exported for convenience)
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEFAULT_RESOLUTION,
    ImageNetIDDataset,
    LabelMapping,
    build_imagenet_loader,
    build_label_mapping,
    build_transform,
    load_imagenet_class_names,
    load_imagenet_wnids,
    synthetic_imagenet_like,
)

logger = logging.getLogger(__name__)

try:  # pragma: no cover - torch is required for dataset objects
    import torch
    from torch.utils.data import Dataset

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Dataset = object  # type: ignore
    _HAS_TORCH = False

try:  # pragma: no cover
    from PIL import Image

    _HAS_PIL = True
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    _HAS_PIL = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OOD_DATASETS: Tuple[str, ...] = (
    "imagenet_v2",
    "imagenet_sketch",
    "imagenet_r",
    "imagenet_a",
    "objectnet",
)

#: Canonical short names used in the paper's tables.
OOD_DISPLAY_NAMES: Dict[str, str] = {
    "imagenet_v2": "ImgN-v2",
    "imagenet_sketch": "ImgN-S",
    "imagenet_r": "ImgN-R",
    "imagenet_a": "ImgN-A",
    "objectnet": "ObjNet",
}

#: Aliases accepted by :func:`get_ood_dataset`.
OOD_ALIASES: Dict[str, str] = {
    "v2": "imagenet_v2",
    "imagenetv2": "imagenet_v2",
    "imagenet-v2": "imagenet_v2",
    "imagenet_v2": "imagenet_v2",
    "s": "imagenet_sketch",
    "sketch": "imagenet_sketch",
    "imagenet-s": "imagenet_sketch",
    "imagenet_s": "imagenet_sketch",
    "imagenet_sketch": "imagenet_sketch",
    "imagenet-sketch": "imagenet_sketch",
    "r": "imagenet_r",
    "rendition": "imagenet_r",
    "imagenet-r": "imagenet_r",
    "imagenet_r": "imagenet_r",
    "imagenet-rendition": "imagenet_r",
    "a": "imagenet_a",
    "adversarial": "imagenet_a",
    "imagenet-a": "imagenet_a",
    "imagenet_a": "imagenet_a",
    "natural-adv-examples": "imagenet_a",
    "objectnet": "objectnet",
    "objnet": "objectnet",
}

# HuggingFace dataset identifiers.
HF_DATASETS: Dict[str, Dict[str, Any]] = {
    "imagenet_sketch": {
        "path": "songweig/imagenet_sketch",
        "config": None,
    },
}

#: ImageNet-v2 MatchedFrequency (Addendum: commit d626240 of vaishaal/ImageNetV2).
IMAGENET_V2_HF_REPO = "vaishaal/ImageNetV2"
IMAGENET_V2_COMMIT = "d626240"
IMAGENET_V2_VARIANTS: Tuple[str, ...] = (
    "matched-frequency",
    "threshold-0.7",
    "top-images",
)
IMAGENET_V2_VARIANT = "matched-frequency"


# ---------------------------------------------------------------------------
# Label orderings shipped with the OOD datasets
# ---------------------------------------------------------------------------

# ImageNet-A: 200 classes; wnids are listed in ``wnids.txt`` (or embedded in the
# `.npy` file names).  Kept in the order used by the official release so that
# dataset index -> wnid is a pure lookup.
IMAGENET_A_WNIDS: Tuple[str, ...] = (
    "n01498041", "n01531178", "n01534433", "n01558993", "n01580077", "n01582220",
    "n01592084", "n01614925", "n01630670", "n01644373", "n01665541", "n01667114",
    "n01667778", "n01685808", "n01689811", "n01692333", "n01693334", "n01694178",
    "n01695060", "n01697457", "n01698640", "n01704323", "n01728572", "n01728920",
    "n01729322", "n01729977", "n01734418", "n01735189", "n01739381", "n01740131",
    "n01744401", "n01748264", "n01749939", "n01755581", "n01756291", "n01768244",
    "n01770081", "n01770393", "n01773157", "n01773549", "n01773797", "n01774384",
    "n01774750", "n01775062", "n01776313", "n01784675", "n01795545", "n01796340",
    "n01798484", "n01806143", "n01806567", "n01807496", "n01817953", "n01818515",
    "n01819313", "n01820546", "n01824575", "n01828970", "n01829413", "n01833805",
    "n01843065", "n01843383", "n01847000", "n01855032", "n01855672", "n01860187",
    "n01871265", "n01872401", "n01873310", "n01877812", "n01882714", "n01883070",
    "n01910747", "n01914609", "n01917289", "n01924916", "n01930112", "n01944390",
    "n01945685", "n01950731", "n01955084", "n01968897", "n01984695", "n01986214",
    "n02002724", "n02006656", "n02007558", "n02009912", "n02011460", "n02012849",
    "n02013706", "n02017213", "n02018207", "n02018795", "n02025239", "n02027492",
    "n02028035", "n02033041", "n02037110", "n02051845", "n02056570", "n02058221",
    "n02066245", "n02071294", "n02074367", "n02077923", "n02085620", "n02085782",
    "n02085936", "n02086079", "n02086240", "n02086646", "n02086910", "n02087046",
    "n02087394", "n02088094", "n02088238", "n02088364", "n02088466", "n02088632",
    "n02089078", "n02089867", "n02089973", "n02090379", "n02090622", "n02090721",
    "n02091032", "n02091134", "n02091244", "n02091467", "n02091635", "n02091831",
    "n02092002", "n02092339", "n02093256", "n02093428", "n02093647", "n02093754",
    "n02093859", "n02093991", "n02094114", "n02094258", "n02094433", "n02095314",
    "n02095570", "n02095889", "n02096051", "n02096177", "n02096294", "n02096437",
    "n02096585", "n02097047", "n02097130", "n02097209", "n02097298", "n02097474",
    "n02097658", "n02098105", "n02098286", "n02098413", "n02099267", "n02099429",
    "n02099601", "n02099712", "n02099849", "n02100236", "n02100583", "n02100735",
    "n02100877", "n02101006", "n02101388", "n02101556", "n02102040", "n02102177",
    "n02102318", "n02102480", "n02102973", "n02104029", "n02104365", "n02105056",
    "n02105162", "n02105251", "n02105412", "n02105505", "n02105641", "n02105855",
    "n02106030", "n02106166", "n02106382", "n02106550", "n02106662", "n02107142",
    "n02107312", "n02107574", "n02107683", "n02107908", "n02108000", "n02108089",
    "n02108422", "n02108551", "n02108915", "n02109047", "n02109525", "n02109961",
    "n02110063", "n02110185", "n02110341", "n02110627", "n02110806", "n02110958",
    "n02111129", "n02111277", "n02111500", "n02111889", "n02112018", "n02112137",
    "n02112350", "n02112706", "n02113023", "n02113186", "n02113624", "n02113712",
    "n02113799", "n02113978", "n02114367", "n02114548", "n02114712", "n02114855",
    "n02115641", "n02115913", "n02116738", "n02117135", "n02119022", "n02119789",
    "n02120079", "n02120505", "n02123045", "n02123159", "n02123394", "n02123597",
    "n02124075", "n02125311", "n02127052", "n02128385", "n02128757", "n02128925",
    "n02129165", "n02129604", "n02130308", "n02132136", "n02133161", "n02134084",
    "n02134418", "n02137549", "n02138441", "n02165105", "n02165456", "n02167151",
    "n02168699", "n02169497", "n02172182", "n02174001", "n02177972", "n02190166",
    "n02206856", "n02219486", "n02226429", "n02229544", "n02231487", "n02233338",
    "n02236044", "n02256656", "n02259212", "n02264363", "n02268443", "n02268853",
    "n02276258", "n02277742", "n02279972", "n02280649", "n02281406", "n02281787",
    "n02317335", "n02319095", "n02321529", "n02325366", "n02326432", "n02328150",
    "n02342885", "n02346627", "n02356798", "n02361337", "n02363005", "n02364673",
    "n02389026", "n02391049", "n02395406", "n02396427", "n02397096", "n02398521",
    "n02403003", "n02408429", "n02410509", "n02412080", "n02415577", "n02417914",
    "n02422106", "n02422699", "n02423022", "n02437312", "n02437616", "n02441942",
    "n02442845", "n02443114", "n02443484", "n02444819", "n02445715", "n02447366",
    "n02451575", "n02454379", "n02457408", "n02480495", "n02480855", "n02481823",
    "n02483362", "n02483708", "n02484975", "n02486261", "n02486410", "n02487347",
    "n02488291", "n02488702", "n02489166", "n02490219", "n02492035", "n02492660",
    "n02493509", "n02493793", "n02494079", "n02497673", "n02500267", "n02504013",
    "n02504458", "n02509815", "n02510455", "n02514041", "n02526121", "n02536864",
    "n02606052", "n02607072", "n02640242", "n02641379", "n02643566", "n02655020",
    "n02666196", "n02667093", "n02669723", "n02672831", "n02676566", "n02687172",
    "n02690373", "n02692877", "n02699494", "n02701002", "n02704792", "n02708093",
    "n02727426", "n02730930", "n02747177", "n02749479", "n02769748", "n02776631",
    "n02777292", "n02782093", "n02783161", "n02786058", "n02787622", "n02788148",
    "n02790996", "n02791124", "n02791270", "n02793495", "n02794156", "n02795169",
    "n02797295", "n02799071", "n02802426", "n02804414", "n02804610", "n02807133",
    "n02808304", "n02808440", "n02814533", "n02814860", "n02815834", "n02817516",
    "n02823428", "n02823750", "n02825657", "n02834397", "n02835271", "n02837789",
    "n02840245", "n02841315", "n02843684", "n02859443", "n02860847", "n02865351",
    "n02869837", "n02870880", "n02871525", "n02877765", "n02879718", "n02883205",
    "n02892201", "n02892767", "n02894605", "n02895154", "n02906734", "n02909870",
    "n02910353", "n02916936", "n02917067", "n02927161", "n02930766", "n02939185",
    "n02948072", "n02950826", "n02951358", "n02951585", "n02963159", "n02965783",
    "n02966193", "n02966687", "n02971356", "n02974003", "n02977058", "n02978881",
    "n02979186", "n02980441", "n02981792", "n02988304", "n02992211", "n02992529",
    "n02999410", "n03000134", "n03000247", "n03000684", "n03014705", "n03016953",
    "n03017168", "n03018349", "n03026506", "n03028079", "n03032252", "n03041632",
    "n03042490", "n03045698", "n03047690", "n03062245", "n03063599", "n03063689",
    "n03065424", "n03075370", "n03085013", "n03089624", "n03095699", "n03100240",
    "n03109150", "n03110669", "n03124043", "n03124170", "n03125729", "n03126707",
    "n03127747", "n03127925", "n03131574", "n03133878", "n03134739", "n03141823",
    "n03146219", "n03160309", "n03179701", "n03180011", "n03187595", "n03188531",
    "n03196217", "n03197337", "n03201208", "n03207743", "n03207941", "n03208938",
    "n03216828", "n03218198", "n03220513", "n03223299", "n03240683", "n03249569",
    "n03250847", "n03255030", "n03259280", "n03271574", "n03272010", "n03272562",
    "n03290653", "n03291819", "n03297495", "n03314780", "n03325584", "n03337140",
    "n03344393", "n03345487", "n03347037", "n03355925", "n03372029", "n03376595",
    "n03379051", "n03384352", "n03388043", "n03388183", "n03388549", "n03393912",
    "n03394916", "n03400231", "n03404251", "n03417042", "n03424325", "n03425413",
    "n03443371", "n03444034", "n03445777", "n03445924", "n03447447", "n03447721",
    "n03450230", "n03452741", "n03457902", "n03459775", "n03461385", "n03467068",
    "n03476684", "n03476991", "n03478589", "n03481172", "n03482405", "n03483316",
    "n03485407", "n03485794", "n03492542", "n03494278", "n03495258", "n03496892",
    "n03498962", "n03527444", "n03529860", "n03530642", "n03532672", "n03534580",
    "n03535780", "n03538406", "n03544143", "n03584254", "n03584829", "n03590841",
    "n03594734", "n03594945", "n03595614", "n03598930", "n03599486", "n03602883",
    "n03617480", "n03623198", "n03627232", "n03630383", "n03633091", "n03637318",
    "n03642806", "n03649909", "n03657121", "n03658185", "n03661043", "n03662601",
    "n03666591", "n03670208", "n03673027", "n03676483", "n03680355", "n03690938",
    "n03691459", "n03692522", "n03697007", "n03706229", "n03709823", "n03710193",
    "n03710637", "n03710721", "n03717622", "n03720891", "n03721384", "n03724870",
    "n03729826", "n03733131", "n03733281", "n03733805", "n03742115", "n03743016",
    "n03759954", "n03761084", "n03763968", "n03764736", "n03769881", "n03770439",
    "n03770679", "n03773504", "n03775071", "n03775546", "n03776460", "n03777568",
    "n03777754", "n03781244", "n03782006", "n03785016", "n03786901", "n03787032",
    "n03788195", "n03788365", "n03791053", "n03792782", "n03792972", "n03793489",
    "n03794056", "n03796401", "n03803284", "n03804744", "n03814639", "n03814906",
    "n03825788", "n03832673", "n03837869", "n03838899", "n03840681", "n03841143",
    "n03843555", "n03854065", "n03857828", "n03866082", "n03868242", "n03868863",
    "n03871628", "n03873416", "n03874293", "n03874599", "n03876231", "n03877472",
    "n03877845", "n03884397", "n03887697", "n03888257", "n03888605", "n03891251",
    "n03891332", "n03895866", "n03899768", "n03902125", "n03903868", "n03908618",
    "n03908714", "n03916031", "n03920288", "n03924679", "n03929660", "n03929855",
    "n03930313", "n03930630", "n03933933", "n03935335", "n03937543", "n03938244",
    "n03942813", "n03944341", "n03947888", "n03950228", "n03954731", "n03956157",
    "n03958227", "n03961711", "n03967562", "n03970156", "n03976467", "n03976657",
    "n03977966", "n03980874", "n03982430", "n03983396", "n03991062", "n03992509",
    "n03995372", "n03998194", "n04004767", "n04005630", "n04008634", "n04009552",
    "n04019541", "n04023962", "n04026417", "n04033901", "n04033995", "n04037443",
    "n04039381", "n04040759", "n04041544", "n04044716", "n04049303", "n04065272",
    "n04067472", "n04069434", "n04070727", "n04074963", "n04081281", "n04086273",
    "n04090263", "n04099969", "n04111531", "n04116512", "n04118538", "n04118776",
    "n04120489", "n04125021", "n04127249", "n04131690", "n04133789", "n04136333",
    "n04141076", "n04141327", "n04141975", "n04146614", "n04147183", "n04149813",
    "n04152593", "n04153751", "n04154565", "n04162706", "n04179913", "n04192698",
    "n04200800", "n04201297", "n04204238", "n04204347", "n04208210", "n04209133",
    "n04209239", "n04228054", "n04229816", "n04235860", "n04238763", "n04239074",
    "n04243546", "n04251144", "n04252077", "n04252225", "n04254120", "n04254680",
    "n04254777", "n04258138", "n04259630", "n04263257", "n04264628", "n04265275",
    "n04266014", "n04270147", "n04273569", "n04275548", "n04277352", "n04285008",
    "n04286575", "n04296562", "n04310018", "n04311004", "n04311174", "n04317175",
    "n04325704", "n04326547", "n04328186", "n04330267", "n04332243", "n04335435",
    "n04336792", "n04344873", "n04346328", "n04347754", "n04350905", "n04355338",
    "n04355933", "n04356056", "n04357314", "n04366367", "n04367480", "n04370456",
    "n04371430", "n04371774", "n04372370", "n04376876", "n04380533", "n04389033",
    "n04392985", "n04398044", "n04399382", "n04404412", "n04409515", "n04417672",
    "n04418357", "n04423845", "n04428191", "n04429376", "n04435653", "n04442312",
    "n04443257", "n04447861", "n04456115", "n04458633", "n04461696", "n04462240",
    "n04465501", "n04467665", "n04476259", "n04479046", "n04482393", "n04483307",
    "n04485082", "n04486054", "n04487081", "n04487394", "n04493381", "n04501370",
    "n04505470", "n04507155", "n04509417", "n04515003", "n04517823", "n04522168",
    "n04523525", "n04525038", "n04525305", "n04532106", "n04532670", "n04536866",
    "n04540053", "n04542943", "n04548280", "n04548362", "n04550184", "n04552348",
    "n04553703", "n04554684", "n04557648", "n04560804", "n04562935", "n04579145",
    "n04579432", "n04584207", "n04589890", "n04590129", "n04591157", "n04591713",
    "n04592741", "n04596742", "n04597913", "n04599235", "n04604644", "n04606251",
    "n04612504", "n04613696", "n06359193", "n06596364", "n06785654", "n06794110",
    "n06874185", "n07248320", "n07565083", "n07579787", "n07583066", "n07584110",
    "n07590611", "n07613480", "n07615774", "n07695742", "n07697313", "n07697537",
    "n07711569", "n07714571", "n07714990", "n07715103", "n07716358", "n07716906",
    "n07717410", "n07717556", "n07718472", "n07718747", "n07720875", "n07730033",
    "n07734744", "n07742313", "n07745940", "n07747607", "n07749582", "n07753113",
    "n07753275", "n07753592", "n07754684", "n07760859", "n07768694", "n07802026",
    "n07831146", "n07836838", "n07860988", "n07871810", "n07873807", "n07875152",
    "n07880968", "n07892512", "n07920052", "n07930864", "n07932039", "n09193705",
    "n09229709", "n09246464", "n09256479", "n09288635", "n09332890", "n09399592",
    "n09421951", "n09428293", "n09468604", "n09472597", "n09835506", "n10148035",
    "n10565667", "n11879895", "n11939491", "n12057211", "n12144580", "n12267677",
    "n12620546", "n12768682", "n12985857", "n12998815", "n13037406", "n13040303",
    "n13044778", "n13052670", "n13054560", "n13133613", "n15075141",
)


# ObjectNet -> ImageNet mapping file name (official release).
OBJECTNET_MAPPING_FILE = "mappings/objectnet_to_imagenet_1k.txt"
OBJECTNET_FOLDER_NAME_FILE = "folder_to_objectnet_label.csv"


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------


class OODDataset(Dataset):  # type: ignore[misc]
    """Thin wrapper adding OOD metadata (name, mapping) to a source dataset.

    ``__getitem__`` returns ``(image_tensor, canonical_label)`` exactly like
    :class:`src.data.imagenet.ImageNetIDDataset`, so OOD and ID loaders are
    interchangeable in the evaluation driver.
    """

    def __init__(
        self,
        source: Any,
        dataset_name: str = "ood",
        label_mapping: Optional[LabelMapping] = None,
        transform: Any = None,
        image_column: str = "image",
        label_column: str = "label",
    ) -> None:
        self.source = source
        self.dataset_name = dataset_name
        self.label_mapping = label_mapping
        self.transform = transform
        self.image_column = image_column
        self.label_column = label_column

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.source)  # type: ignore[arg-type]

    def _resolve(self, idx: int) -> Tuple[Any, int]:
        record = self.source[idx]
        if isinstance(record, dict):
            image = record[self.image_column]
            label = record[self.label_column]
        elif isinstance(record, (tuple, list)) and len(record) >= 2:
            image, label = record[0], record[1]
        else:  # pragma: no cover - defensive
            raise TypeError(
                f"Unsupported OOD record type {type(record)}; expected dict or (image, label)."
            )
        return image, int(label)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        image, label = self._resolve(idx)
        if self.label_mapping is not None:
            label = self.label_mapping.remap(label)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def num_classes(self) -> int:
        if self.label_mapping is not None:
            return self.label_mapping.num_classes
        return 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrapped_loader(loader_cls: Any, **kwargs: Any) -> Any:
    return loader_cls(**kwargs)


def subsample(dataset: Any, num_samples: Optional[int], seed: int = 0) -> Any:
    """Deterministic subsample used for smoke tests / fast caching."""
    if num_samples is None or num_samples >= len(dataset):  # type: ignore[arg-type]
        return dataset
    from .imagenet import subset_dataset

    return subset_dataset(dataset, num_samples, seed=seed)


def build_ood_transform(
    resolution: int = DEFAULT_RESOLUTION,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    train: bool = False,
) -> Any:
    """Alias of :func:`src.data.imagenet.build_transform` (identical preprocessing)."""
    return build_transform(resolution=resolution, mean=mean, std=std, train=train)


# ---------------------------------------------------------------------------
# ImageNet-v2 (MatchedFrequency)
# ---------------------------------------------------------------------------


def _find_imagenet_v2_dir(root: str, variant: str = IMAGENET_V2_VARIANT) -> Optional[str]:
    """Locate the ImageNet-v2 ``MatchedFrequency`` directory under ``root``."""
    candidates = [
        "imagenetv2-matched-frequency-format-val",
        "imagenetv2-matched-frequency",
        "matched-frequency",
        "imagenetv2",
        ".",
    ]
    for name in candidates:
        candidate = os.path.join(root, name)
        if os.path.isdir(candidate):
            return candidate
    return None


def load_imagenet_v2_local(root: str, variant: str = IMAGENET_V2_VARIANT) -> Any:
    """Load ImageNet-v2 from a local ``imagenetv2-*-format-val`` directory.

    ImageNet-v2 folder names are ImageNet class *indices* (0..999), matching the
    canonical ordering, so an :class:`~torchvision.datasets.ImageFolder` already
    yields aligned labels.  A per-class ``classname.txt``/``imagelist.txt`` may
    be present; the folder index is sufficient.
    """
    from torchvision.datasets import ImageFolder

    folder = _find_imagenet_v2_dir(root, variant)
    if folder is None:
        folder = root
    return ImageFolder(root=folder)


def load_imagenet_v2_hf(
    variant: str = IMAGENET_V2_VARIANT,
    cache_dir: Optional[str] = None,
    commit: str = IMAGENET_V2_COMMIT,
    trust_remote_code: bool = True,
) -> Any:
    """Load ImageNet-v2 from the ``vaishaal/ImageNetV2`` HF repo (Addendum)."""
    from datasets import load_dataset

    logger.info(
        "Loading ImageNet-v2 (%s) from HF %s@%s", variant, IMAGENET_V2_HF_REPO, commit
    )
    try:
        return load_dataset(
            IMAGENET_V2_HF_REPO,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            revision=commit,
        )
    except TypeError:  # older datasets versions lack `revision`/`trust_remote_code`
        return load_dataset(IMAGENET_V2_HF_REPO, cache_dir=cache_dir)


def build_imagenet_v2_dataset(
    root: Optional[str] = None,
    variant: str = IMAGENET_V2_VARIANT,
    transform: Any = None,
    use_hf: bool = True,
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    seed: int = 0,
    allow_synthetic: bool = False,
) -> OODDataset:
    """ImageNet-v2 (MatchedFrequency) loader with canonical label alignment."""
    transform = transform if transform is not None else build_ood_transform()
    source = None

    if root and os.path.isdir(root):
        try:
            source = load_imagenet_v2_local(root, variant=variant)
        except Exception as exc:  # pragma: no cover - depends on local files
            logger.warning("Local ImageNet-v2 load failed (%s); trying HuggingFace.", exc)

    if source is None and use_hf:
        try:
            hf = load_imagenet_v2_hf(variant=variant, cache_dir=cache_dir)
            split = "train" if "train" in hf else list(hf.keys())[0]
            source = hf[split]
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("HF ImageNet-v2 unavailable (%s).", exc)

    if source is None:
        if not allow_synthetic:
            raise FileNotFoundError(
                "ImageNet-v2 not found. Provide `root` or enable `use_hf`/`allow_synthetic`."
            )
        source = synthetic_imagenet_like(num_samples=max_samples or 64)

    dataset = OODDataset(source, dataset_name="imagenet_v2", transform=transform)
    if max_samples:
        dataset = subsample(dataset, max_samples, seed=seed)
    else:
        dataset = OODDataset(source, dataset_name="imagenet_v2", transform=transform)
    return dataset


# ---------------------------------------------------------------------------
# ImageNet-S (Sketch)
# ---------------------------------------------------------------------------


def load_imagenet_sketch_local(root: str) -> Any:
    """ImageNet-Sketch from a local ``sketch/`` directory (1000 subfolders)."""
    from torchvision.datasets import ImageFolder

    for candidate in ("sketch", "imagenet-sketch", "imagenet_sketch", "."):
        path = os.path.join(root, candidate)
        if os.path.isdir(path):
            return ImageFolder(root=path)
    return ImageFolder(root=root)


def load_imagenet_sketch_hf(
    cache_dir: Optional[str] = None, trust_remote_code: bool = True
) -> Any:
    """ImageNet-Sketch from ``songweig/imagenet_sketch`` (Addendum)."""
    from datasets import load_dataset

    spec = HF_DATASETS["imagenet_sketch"]
    logger.info("Loading ImageNet-Sketch from HF %s", spec["path"])
    kwargs: Dict[str, Any] = {"cache_dir": cache_dir, "trust_remote_code": trust_remote_code}
    if spec["config"]:
        kwargs["name"] = spec["config"]
    try:
        return load_dataset(spec["path"], **kwargs)
    except TypeError:
        kwargs.pop("trust_remote_code", None)
        return load_dataset(spec["path"], **kwargs)


def build_imagenet_sketch_dataset(
    root: Optional[str] = None,
    transform: Any = None,
    use_hf: bool = True,
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    seed: int = 0,
    allow_synthetic: bool = False,
) -> OODDataset:
    """ImageNet-S loader. Labels follow ImageNet ordering (ImageFolder of wnids)."""
    transform = transform if transform is not None else build_ood_transform()
    source = None

    if root and os.path.isdir(root):
        try:
            source = load_imagenet_sketch_local(root)
        except Exception as exc:  # pragma: no cover
            logger.warning("Local ImageNet-S load failed (%s); trying HuggingFace.", exc)

    if source is None and use_hf:
        try:
            hf = load_imagenet_sketch_hf(cache_dir=cache_dir)
            split = "test" if "test" in hf else list(hf.keys())[0]
            source = hf[split]
        except Exception as exc:  # pragma: no cover
            logger.warning("HF ImageNet-Sketch unavailable (%s).", exc)

    if source is None:
        if not allow_synthetic:
            raise FileNotFoundError(
                "ImageNet-Sketch not found. Provide `root` or enable `use_hf`/`allow_synthetic`."
            )
        source = synthetic_imagenet_like(num_samples=max_samples or 64)

    dataset = OODDataset(source, dataset_name="imagenet_sketch", transform=transform)
    if max_samples:
        dataset = subsample(dataset, max_samples, seed=seed)
    return dataset


# ---------------------------------------------------------------------------
# ImageNet-R / ImageNet-A (wnid-ordered)
# ---------------------------------------------------------------------------


def load_wnid_ordered_folder(root: str) -> Any:
    """ImageFolder whose class dirs are synset ids (ImageNet-A / ImageNet-R)."""
    from torchvision.datasets import ImageFolder

    return ImageFolder(root=root)


def build_wnid_label_mapping(class_names: Optional[Sequence[str]] = None) -> LabelMapping:
    """Build a :class:`LabelMapping` for datasets whose dirs are ImageNet wnids."""
    wnids = load_imagenet_wnids()
    names = list(class_names) if class_names is not None else load_imagenet_class_names()
    if len(names) != len(wnids):
        names = [f"class_{i}" for i in range(len(wnids))]
    return LabelMapping(index_to_wnid=list(wnids), index_to_name=names)


class WnidOrderedDataset(Dataset):  # type: ignore[misc]
    """Wraps an ImageFolder whose class names are ImageNet synset ids.

    The wrapping dataset's class order (``classes``) is looked up against the
    canonical wnid list to produce per-index canonical labels, so that returned
    labels align with the WordNet class indices.
    """

    def __init__(
        self,
        folder: Any,
        dataset_name: str = "wnid_ood",
        transform: Any = None,
        canonical_wnids: Optional[Sequence[str]] = None,
        label_mapping: Optional[LabelMapping] = None,
    ) -> None:
        self.folder = folder
        self.dataset_name = dataset_name
        self.transform = transform
        self.canonical_wnids = list(canonical_wnids) if canonical_wnids else load_imagenet_wnids()
        self.wnid_to_index = {w: i for i, w in enumerate(self.canonical_wnids)}

        classes = list(getattr(folder, "classes", []))
        local_to_canonical: List[int] = []
        for name in classes:
            wnid = os.path.basename(str(name))
            local_to_canonical.append(self.wnid_to_index.get(wnid, -1))
        n_unmapped = sum(1 for v in local_to_canonical if v < 0)
        if n_unmapped:
            logger.warning(
                "%s: %d/%d classes could not be mapped to ImageNet wnids (using -1).",
                dataset_name,
                n_unmapped,
                len(local_to_canonical),
            )
        self.local_to_canonical = local_to_canonical
        self.classes = classes
        self.label_mapping = label_mapping

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.folder)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        image, local_label = self.folder[idx]
        label = self.local_to_canonical[int(local_label)]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def num_classes(self) -> int:
        return len(self.canonical_wnids)


def build_imagenet_r_dataset(
    root: Optional[str] = None,
    transform: Any = None,
    max_samples: Optional[int] = None,
    seed: int = 0,
    allow_synthetic: bool = False,
    **_: Any,
) -> OODDataset:
    """ImageNet-R loader (https://github.com/hendrycks/imagenet-r).

    Expects the ``imagenetr/`` (or ``imagenet-r/``) folder with synset-id class
    directories.
    """
    transform = transform if transform is not None else build_ood_transform()
    folder = None
    if root and os.path.isdir(root):
        for candidate in ("imagenetr", "imagenet-r", "imagenet_r", "."):
            path = os.path.join(root, candidate)
            dirs = None
            if os.path.isdir(path):
                subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                if any(d.startswith("n") and len(d) == 9 for d in subdirs):
                    dirs = path
                    break
        if dirs is not None:
            folder = load_wnid_ordered_folder(dirs)

    if folder is None:
        if not allow_synthetic:
            raise FileNotFoundError(
                "ImageNet-R not found. Download from https://github.com/hendrycks/imagenet-r "
                "and provide `root`."
            )
        source = synthetic_imagenet_like(num_samples=max_samples or 64)
        return OODDataset(source, dataset_name="imagenet_r", transform=transform)

    dataset = WnidOrderedDataset(folder, dataset_name="imagenet_r", transform=transform)
    if max_samples:
        dataset = subsample(dataset, max_samples, seed=seed)
    return dataset


def build_imagenet_a_dataset(
    root: Optional[str] = None,
    transform: Any = None,
    max_samples: Optional[int] = None,
    seed: int = 0,
    allow_synthetic: bool = False,
    npy_path: Optional[str] = None,
    **_: Any,
) -> OODDataset:
    """ImageNet-A loader (https://github.com/hendrycks/natural-adv-examples).

    Supports both the official ``.npy`` format (``x_test.npy``/``y_test.npy``)
    and the common extracted ``imagenet-a/`` folder of synset directories.
    """
    transform = transform if transform is not None else build_ood_transform()

    # Preferred: the official .npy arrays.
    if npy_path is None and root:
        for name in ("imagenet-a.npy", "imagenet_a.npy"):
            candidate = os.path.join(root, name)
            if os.path.isfile(candidate):
                npy_path = candidate
                break
    if npy_path and os.path.isfile(npy_path):
        dataset = NpyClassDataset(
            npy_path,
            canonical_wnids=load_imagenet_wnids(),
            dataset_name="imagenet_a",
            transform=transform,
        )
        if max_samples:
            dataset = subsample(dataset, max_samples, seed=seed)
        return dataset

    folder = None
    if root and os.path.isdir(root):
        for candidate in ("imagenet-a", "imagenet_a", "imagenetadv", "."):
            path = os.path.join(root, candidate)
            if os.path.isdir(path):
                subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                if any(d.startswith("n") and len(d) == 9 for d in subdirs):
                    folder = path
                    break
    if folder is not None:
        dataset = WnidOrderedDataset(folder, dataset_name="imagenet_a", transform=transform)
        if max_samples:
            dataset = subsample(dataset, max_samples, seed=seed)
        return dataset

    if not allow_synthetic:
        raise FileNotFoundError(
            "ImageNet-A not found. Download from "
            "https://github.com/hendrycks/natural-adv-examples and provide `root`."
        )
    source = synthetic_imagenet_like(num_samples=max_samples or 64)
    return OODDataset(source, dataset_name="imagenet_a", transform=transform)


class NpyClassDataset(Dataset):  # type: ignore[misc]
    """ImageNet-A ``.npy`` file (``(N, 2, 224, 224, 3)`` uint8 or object array).

    ``y`` holds ImageNet *class indices* (0..999) for the 200 ImageNet-A classes
    in their sorted wnid order; when the file stores wnids instead, pass them via
    ``wnids``.
    """

    def __init__(
        self,
        path: str,
        canonical_wnids: Sequence[str],
        dataset_name: str = "npy",
        transform: Any = None,
        wnids: Optional[Sequence[str]] = None,
    ) -> None:
        import numpy as np

        self.path = path
        self.dataset_name = dataset_name
        self.transform = transform
        self.canonical_wnids = list(canonical_wnids)
        self.wnid_to_index = {w: i for i, w in enumerate(self.canonical_wnids)}

        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):  # pragma: no cover
            keys = list(data.keys())
            self.images = data[keys[0]]
            self.labels = data[keys[1]] if len(keys) > 1 else None
        else:
            arr = np.asarray(data)
            if arr.ndim == 2 and arr.shape[-1] == 2:  # (N, 2) object array
                images = [row[0] for row in arr]
                self.images = np.stack(images) if images and isinstance(images[0], np.ndarray) else images
                self.labels = np.asarray([int(row[1]) for row in arr])
            else:
                raise ValueError(
                    f"Unsupported ImageNet-A npy layout with shape {arr.shape}."
                )
        if wnids is not None:
            self.labels = np.asarray([self.wnid_to_index.get(w, -1) for w in wnids])

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        image = self.images[idx]
        if not _HAS_PIL:  # pragma: no cover
            raise ImportError("PIL is required to load ImageNet-A images.")
        if not isinstance(image, Image.Image):
            import numpy as np

            image = Image.fromarray(np.asarray(image).astype("uint8"))
        label = int(self.labels[idx]) if self.labels is not None else -1
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# ObjectNet
# ---------------------------------------------------------------------------


def parse_objectnet_mapping(
    mapping_path: str,
) -> Tuple[List[int], List[List[int]]]:
    """Parse ``objectnet_to_imagenet_1k.txt``.

    Returns ``(objectnet_label_ids, imagenet_label_ids_per_objectnet_label)``.
    """
    objectnet_label_ids: List[int] = []
    imagenet_label_ids: List[List[int]] = []
    with open(mapping_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                obj_id = int(parts[0])
                im_ids = [int(p) for p in parts[1:]]
            except ValueError:
                continue
            objectnet_label_ids.append(obj_id)
            imagenet_label_ids.append(im_ids)
    return objectnet_label_ids, imagenet_label_ids


class ObjectNetDataset(Dataset):  # type: ignore[misc]
    """ObjectNet wrapper mapping its 313 folders onto the ImageNet taxonomy.

    For each ObjectNet folder, the official mapping lists the compatible
    ImageNet classes.  Following the paper's setup we assign every image to the
    *first* mapped ImageNet class (deterministic) so a single canonical label is
    available for both accuracy and LCA computation.
    """

    def __init__(
        self,
        folder: Any,
        objectnet_label_ids: Sequence[int],
        imagenet_label_ids: Sequence[Sequence[int]],
        dataset_name: str = "objectnet",
        transform: Any = None,
        canonical_wnids: Optional[Sequence[str]] = None,
    ) -> None:
        self.folder = folder
        self.dataset_name = dataset_name
        self.transform = transform
        self.canonical_wnids = list(canonical_wnids) if canonical_wnids else load_imagenet_wnids()

        self.mapping: Dict[int, List[int]] = {
            int(k): [int(v) for v in vals]
            for k, vals in zip(objectnet_label_ids, imagenet_label_ids)
        }
        classes = list(getattr(folder, "classes", []))
        self.classes = classes

        # Resolve each ObjectNet folder to a canonical ImageNet index.
        local_to_canonical: List[int] = []
        for i, _name in enumerate(classes):
            candidates = self.mapping.get(i, [])
            local_to_canonical.append(int(candidates[0]) if candidates else -1)
        n_unmapped = sum(1 for v in local_to_canonical if v < 0)
        if n_unmapped:
            logger.warning(
                "ObjectNet: %d/%d folders lacked a mapped ImageNet class.", n_unmapped, len(classes)
            )
        self.local_to_canonical = local_to_canonical

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.folder)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        image, local = self.folder[idx]
        label = self.local_to_canonical[int(local)]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def num_classes(self) -> int:
        return len(self.canonical_wnids)


def build_objectnet_dataset(
    root: Optional[str] = None,
    transform: Any = None,
    max_samples: Optional[int] = None,
    seed: int = 0,
    allow_synthetic: bool = False,
    mapping_path: Optional[str] = None,
    **_: Any,
) -> OODDataset:
    """ObjectNet loader (https://objectnet.dev/)."""
    from torchvision.datasets import ImageFolder

    transform = transform if transform is not None else build_ood_transform()

    folder_path = None
    if root and os.path.isdir(root):
        for candidate in ("objectnet", "images", "."):
            path = os.path.join(root, candidate)
            if os.path.isdir(path):
                subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                if len(subdirs) > 50:  # ObjectNet has 313 folders
                    folder_path = path
                    break

    if folder_path is None:
        if not allow_synthetic:
            raise FileNotFoundError(
                "ObjectNet not found. Download from https://objectnet.dev/ and provide `root`."
            )
        source = synthetic_imagenet_like(num_samples=max_samples or 64)
        return OODDataset(source, dataset_name="objectnet", transform=transform)

    if mapping_path is None:
        for candidate in (
            os.path.join(root or "", OBJECTNET_MAPPING_FILE),
            os.path.join(root or "", "objectnet_to_imagenet_1k.txt"),
            os.path.join(root or "", "mappings", "objectnet_to_imagenet_1k.txt"),
        ):
            if os.path.isfile(candidate):
                mapping_path = candidate
                break

    if mapping_path is None or not os.path.isfile(mapping_path):
        logger.warning(
            "ObjectNet mapping file not found; treating folder order as ImageNet order."
        )
        folder = ImageFolder(root=folder_path)
        dataset = WnidOrderedDataset(folder, dataset_name="objectnet", transform=transform)
    else:
        obj_ids, im_ids = parse_objectnet_mapping(mapping_path)
        folder = ImageFolder(root=folder_path)
        dataset = ObjectNetDataset(folder, obj_ids, im_ids, transform=transform)

    if max_samples:
        dataset = subsample(dataset, max_samples, seed=seed)
    return dataset


# ---------------------------------------------------------------------------
# Registry / top-level entry points
# ---------------------------------------------------------------------------

BUILDERS: Dict[str, Any] = {
    "imagenet_v2": build_imagenet_v2_dataset,
    "imagenet_sketch": build_imagenet_sketch_dataset,
    "imagenet_s": build_imagenet_sketch_dataset,
    "imagenet_r": build_imagenet_r_dataset,
    "imagenet_a": build_imagenet_a_dataset,
    "objectnet": build_objectnet_dataset,
}


def normalize_ood_name(name: str) -> str:
    """Map aliases (``v2``, ``S``, ``imagenet-r`` ...) to canonical keys."""
    key = str(name).strip().lower()
    return OOD_ALIASES.get(key, key)


def build_ood_dataset(
    name: str,
    root: Optional[str] = None,
    transform: Any = None,
    resolution: int = DEFAULT_RESOLUTION,
    max_samples: Optional[int] = None,
    seed: int = 0,
    use_hf: bool = True,
    cache_dir: Optional[str] = None,
    allow_synthetic: bool = False,
    **kwargs: Any,
) -> Any:
    """Build one OOD dataset by name with a canonical ImageNet label ordering."""
    canonical = normalize_ood_name(name)
    if canonical not in BUILDERS:
        raise KeyError(f"Unknown OOD dataset '{name}'. Known: {sorted(BUILDERS)}")

    if transform is None:
        transform = build_ood_transform(resolution=resolution)

    builder = BUILDERS[canonical]
    builder_kwargs: Dict[str, Any] = {
        "root": root,
        "transform": transform,
        "max_samples": max_samples,
        "seed": seed,
        "allow_synthetic": allow_synthetic,
    }
    if canonical in ("imagenet_v2", "imagenet_sketch"):
        builder_kwargs["use_hf"] = use_hf
        builder_kwargs["cache_dir"] = cache_dir
    builder_kwargs.update(kwargs)
    return builder(**builder_kwargs)


def build_all_ood_datasets(
    roots: Optional[Dict[str, str]] = None,
    resolution: int = DEFAULT_RESOLUTION,
    max_samples: Optional[int] = None,
    seed: int = 0,
    use_hf: bool = True,
    cache_dir: Optional[str] = None,
    allow_synthetic: bool = False,
    names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build every OOD dataset in :data:`DEFAULT_OOD_DATASETS`.

    Failures are logged and skipped unless ``allow_synthetic`` is set, so a
    partially downloaded data directory still produces partial results.
    """
    roots = roots or {}
    names = names or DEFAULT_OOD_DATASETS
    transform = build_ood_transform(resolution=resolution)
    datasets: Dict[str, Any] = {}
    for name in names:
        canonical = normalize_ood_name(name)
        root = roots.get(canonical, roots.get(name))
        try:
            datasets[canonical] = build_ood_dataset(
                canonical,
                root=root,
                transform=transform,
                max_samples=max_samples,
                seed=seed,
                use_hf=use_hf,
                cache_dir=cache_dir,
                allow_synthetic=allow_synthetic,
            )
        except Exception as exc:  # pragma: no cover - depends on local data
            if allow_synthetic:
                logger.warning("%s unavailable (%s); using synthetic fallback.", canonical, exc)
                datasets[canonical] = OODDataset(
                    synthetic_imagenet_like(transform=transform, num_samples=max_samples or 64),
                    dataset_name=canonical,
                    transform=None,
                )
            else:
                logger.warning("Skipping %s: %s", canonical, exc)
    return datasets


def display_name(name: str) -> str:
    """Paper-style short label (``ImgN-v2``, ``ObjectNet`` ...)."""
    return OOD_DISPLAY_NAMES.get(normalize_ood_name(name), name)


__all__ = [
    "DEFAULT_OOD_DATASETS",
    "OOD_ALIASES",
    "OOD_DISPLAY_NAMES",
    "HF_DATASETS",
    "IMAGENET_V2_HF_REPO",
    "IMAGENET_V2_COMMIT",
    "IMAGENET_V2_VARIANT",
    "IMAGENET_V2_VARIANTS",
    "IMAGENET_A_WNIDS",
    "OODDataset",
    "WnidOrderedDataset",
    "NpyClassDataset",
    "ObjectNetDataset",
    "build_ood_transform",
    "build_imagenet_v2_dataset",
    "build_imagenet_sketch_dataset",
    "build_imagenet_r_dataset",
    "build_imagenet_a_dataset",
    "build_objectnet_dataset",
    "build_ood_dataset",
    "build_all_ood_datasets",
    "load_imagenet_v2_local",
    "load_imagenet_v2_hf",
    "load_imagenet_sketch_local",
    "load_imagenet_sketch_hf",
    "parse_objectnet_mapping",
    "normalize_ood_name",
    "display_name",
    "subsample",
    "build_wnid_label_mapping",
]
