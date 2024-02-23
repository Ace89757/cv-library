# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import logging

from mmengine.logging import print_log


def heading_line(log, logger=None):
    if logger is None:
        print_log('{:-^100}'.format(f' {log} '), logger='current', level=logging.INFO)
    else:
        logger.info('{:-^100}'.format(f' {log} '))

