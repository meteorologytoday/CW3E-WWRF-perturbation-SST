#!/bin/bash


subgroupLabel() {
    local subgroup=$1
    local subgroup_str=""
    if [ "$subgroup" = "BLANK" ] ; then
        subgroup_str="_BLANK"
    else
        subgroup_str="_${subgroup}"
    fi
 
    echo "$subgroup_str"
}
