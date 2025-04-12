#import "template.typ": template
#import "cfg.typ": cfg

#show: body => template(cfg: cfg, body)

#set math.equation(numbering: "(1)")

#include "body.typ"
